// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shard plan generator for tensor parallelism.
//!
//! Given a model architecture and configuration, produces a [`ModelShardPlan`]
//! that describes how every weight tensor should be sharded across TP ranks.
//!
//! Architecture-specific builders handle naming conventions while the core
//! sharding rules (column-parallel for Q/K/V/gate/up, row-parallel for O/down)
//! remain consistent across all transformer architectures.
//!
//! Used by: model loading pipeline (weight sharding), server startup

use anyhow::Result;

use super::config::{EmbeddingMode, MoeShardMode, ShardConfig};
use super::shard_strategy::{CommPattern, LayerShardPlan, ModelShardPlan, ShardStrategy};

/// Generate a shard plan for the given model architecture.
///
/// # Arguments
/// * `architecture` - Model architecture name (e.g., "llama", "qwen2", "mixtral")
/// * `num_layers` - Number of transformer/SSM layers in the model
/// * `config` - User-configurable sharding options
///
/// # Returns
/// A [`ModelShardPlan`] describing how to shard every weight tensor.
pub fn generate_shard_plan(
    architecture: &str,
    num_layers: usize,
    config: &ShardConfig,
) -> Result<ModelShardPlan> {
    config.validate()?;

    if config.tp_size == 1 {
        return Ok(build_replicated_plan(architecture, num_layers));
    }

    let arch_lower = architecture.to_ascii_lowercase();
    let arch_key = arch_lower.as_str();

    match arch_key {
        // Llama family (Llama 1/2/3, Mistral, Yi, TinyLlama, Vicuna, etc.)
        "llama" | "llama3" | "mistral" | "yi" | "tinyllama" | "vicuna" => {
            build_llama_plan(num_layers, config)
        }

        // Llama 4 (MoE with interleaved grouped query attention)
        "llama4" => build_llama4_plan(num_layers, config),

        // Qwen family
        "qwen2" | "qwen2.5" | "qwen3" => build_qwen_plan(num_layers, config),
        "qwen3_5" | "qwen3.5" | "qwen3_5_text" | "qwen3.5_text" => {
            build_qwen35_plan(num_layers, config)
        }

        // Qwen MoE variants
        "qwen2_moe" | "qwen3_moe" | "qwen3_5_moe" | "qwen3.5_moe" => {
            build_qwen_moe_plan(num_layers, config)
        }

        // Gemma family
        "gemma" | "gemma2" | "gemma3" | "gemma3n" => build_gemma_plan(num_layers, config),
        "gemma4" | "gemma4_text" => build_gemma4_plan(num_layers, config),

        // Phi family
        "phi" | "phi3" | "phi3small" | "phi4mm" => build_phi_plan(num_layers, config),

        // Phi MoE
        "phimoe" => build_phi_moe_plan(num_layers, config),

        // Mixtral (MoE)
        "mixtral" => build_mixtral_plan(num_layers, config),

        // Ministral3 / Mistral3
        "ministral3" | "mistral3" => build_llama_plan(num_layers, config),

        // DeepSeek family (MLA + MoE for v2/v3)
        "deepseek" => build_deepseek_dense_plan(num_layers, config),
        "deepseek_v2" | "deepseek_v3" | "deepseek_v32" => {
            build_deepseek_moe_plan(num_layers, config)
        }

        // Cohere family
        "cohere" | "cohere2" => build_cohere_plan(num_layers, config),

        // StarCoder2
        "starcoder2" => build_starcoder2_plan(num_layers, config),

        // OLMo family
        "olmo" | "olmo2" | "olmo3" => build_llama_plan(num_layers, config),

        // OLMoE
        "olmoe" => build_olmoe_plan(num_layers, config),

        // InternLM
        "internlm2" | "internlm3" => build_llama_plan(num_layers, config),

        // ExaOne family
        "exaone" | "exaone4" => build_llama_plan(num_layers, config),
        "exaone_moe" => build_generic_moe_plan(num_layers, config, "exaone_moe"),

        // GLM family
        "glm4" => build_llama_plan(num_layers, config),
        "glm4_moe" | "glm4_moe_lite" | "glm_moe_dsa" => {
            build_generic_moe_plan(num_layers, config, arch_key)
        }

        // Chinese model families
        "baichuan" => build_llama_plan(num_layers, config),
        "ernie4_5" => build_llama_plan(num_layers, config),
        "ernie4_5_moe" => build_generic_moe_plan(num_layers, config, "ernie4_5_moe"),
        "hunyuan_moe" => build_generic_moe_plan(num_layers, config, "hunyuan_moe"),
        "hunyuan_v1_dense" => build_llama_plan(num_layers, config),
        "mimo" => build_llama_plan(num_layers, config),

        // Korean model families
        "solar_open" => build_llama_plan(num_layers, config),

        // MiniMax (large MoE)
        "minimax" => build_generic_moe_plan(num_layers, config, "minimax"),

        // MiniCPM
        "minicpm" | "minicpm3" => build_llama_plan(num_layers, config),

        // Other dense transformers
        "stablelm" | "smollm3" | "nemotron" | "gpt_oss" | "step3p5" => {
            build_llama_plan(num_layers, config)
        }

        // SSM/hybrid models: replicate for now (TP for SSM is non-trivial)
        "mamba"
        | "mamba2"
        | "falcon_mamba"
        | "jamba"
        | "nemotron_h"
        | "nemotron_nas"
        | "rwkv7"
        | "recurrent_gemma"
        | "kimi_linear"
        | "kimi_k3"
        | "longcat_flash"
        | "longcat_flash_ngram"
        | "qwen3_next" => Ok(build_replicated_plan(architecture, num_layers)),

        _ => {
            // Fallback: use generic transformer plan
            build_generic_transformer_plan(num_layers, config, architecture)
        }
    }
}

// ---------------------------------------------------------------------------
// Internal plan builders
// ---------------------------------------------------------------------------

/// Build a fully-replicated plan (tp_size=1 or unsupported architecture).
fn build_replicated_plan(architecture: &str, num_layers: usize) -> ModelShardPlan {
    ModelShardPlan {
        tp_size: 1,
        num_layers,
        layer_plans: Vec::new(),
        embedding_strategy: ShardStrategy::Replicated,
        lm_head_strategy: ShardStrategy::Replicated,
        architecture: architecture.to_string(),
    }
}

fn resolve_embedding_strategy(mode: EmbeddingMode) -> ShardStrategy {
    match mode {
        EmbeddingMode::VocabParallel => ShardStrategy::VocabParallel,
        EmbeddingMode::Replicated => ShardStrategy::Replicated,
    }
}

/// Standard attention shard plans for a "model.layers.{}" prefix.
///
/// Used by: Llama, Qwen, Gemma, Phi, Cohere, OLMo, InternLM, ExaOne, Baichuan,
/// StarCoder2, StableLM, MiniCPM, Nemotron, and most other dense transformers.
fn attention_plans(prefix: &str) -> Vec<LayerShardPlan> {
    vec![
        // Q projection: column-parallel (shard heads across ranks)
        LayerShardPlan {
            weight_pattern: format!("{prefix}.self_attn.q_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        // K projection: column-parallel
        LayerShardPlan {
            weight_pattern: format!("{prefix}.self_attn.k_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        // V projection: column-parallel
        LayerShardPlan {
            weight_pattern: format!("{prefix}.self_attn.v_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        // O projection: row-parallel (combine partial head outputs)
        LayerShardPlan {
            weight_pattern: format!("{prefix}.self_attn.o_proj.weight"),
            strategy: ShardStrategy::RowParallel,
            shard_axis: 1,
            comm_pattern: CommPattern::AllReduce,
        },
    ]
}

/// Standard FFN shard plans for gate/up/down projections.
///
/// Used by: Llama, Qwen, Gemma, Phi, OLMo, InternLM, ExaOne, Baichuan, etc.
fn ffn_plans(prefix: &str) -> Vec<LayerShardPlan> {
    vec![
        // Gate projection: column-parallel
        LayerShardPlan {
            weight_pattern: format!("{prefix}.mlp.gate_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        // Up projection: column-parallel
        LayerShardPlan {
            weight_pattern: format!("{prefix}.mlp.up_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        // Down projection: row-parallel
        LayerShardPlan {
            weight_pattern: format!("{prefix}.mlp.down_proj.weight"),
            strategy: ShardStrategy::RowParallel,
            shard_axis: 1,
            comm_pattern: CommPattern::AllReduce,
        },
    ]
}

/// MoE expert shard plans for a model with `block_sparse_moe` or `moe` submodule.
fn moe_expert_plans(
    prefix: &str,
    moe_submodule: &str,
    config: &ShardConfig,
) -> Vec<LayerShardPlan> {
    match config.moe_mode {
        MoeShardMode::ExpertParallel => {
            // Entire experts are assigned to ranks. The gate is replicated.
            vec![
                LayerShardPlan {
                    weight_pattern: format!("{prefix}.{moe_submodule}.gate.weight"),
                    strategy: ShardStrategy::Replicated,
                    shard_axis: 0,
                    comm_pattern: CommPattern::None,
                },
                LayerShardPlan {
                    weight_pattern: format!("{prefix}.{moe_submodule}.experts"),
                    strategy: ShardStrategy::ExpertParallel,
                    shard_axis: 0,
                    comm_pattern: CommPattern::AllReduce,
                },
            ]
        }
        MoeShardMode::WithinExpert => {
            // Each expert's FFN is sharded like a regular FFN.
            vec![
                LayerShardPlan {
                    weight_pattern: format!("{prefix}.{moe_submodule}.gate.weight"),
                    strategy: ShardStrategy::Replicated,
                    shard_axis: 0,
                    comm_pattern: CommPattern::None,
                },
                LayerShardPlan {
                    weight_pattern: format!("{prefix}.{moe_submodule}.experts.*.gate_proj.weight"),
                    strategy: ShardStrategy::ColumnParallel,
                    shard_axis: 0,
                    comm_pattern: CommPattern::None,
                },
                LayerShardPlan {
                    weight_pattern: format!("{prefix}.{moe_submodule}.experts.*.up_proj.weight"),
                    strategy: ShardStrategy::ColumnParallel,
                    shard_axis: 0,
                    comm_pattern: CommPattern::None,
                },
                LayerShardPlan {
                    weight_pattern: format!("{prefix}.{moe_submodule}.experts.*.down_proj.weight"),
                    strategy: ShardStrategy::RowParallel,
                    shard_axis: 1,
                    comm_pattern: CommPattern::AllReduce,
                },
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture-specific builders
// ---------------------------------------------------------------------------

/// Llama-family (Llama 1/2/3, Mistral, Yi, etc.): standard prefix `model.layers.{}`.
fn build_llama_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "llama".to_string(),
    })
}

/// Llama 4 (MoE with interleaved grouped query attention).
fn build_llama4_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    // Llama 4 has both dense FFN layers and MoE layers (interleaved).
    // Dense FFN
    plans.extend(ffn_plans(prefix));
    // MoE layers use `feed_forward` submodule
    plans.extend(moe_expert_plans(prefix, "feed_forward", config));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "llama4".to_string(),
    })
}

/// Qwen 2/2.5/3/3.5 dense models.
fn build_qwen_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "qwen".to_string(),
    })
}

/// Qwen 3.5 dense models: hybrid full-attention + GatedDeltaNet linear attention.
fn build_qwen35_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    plans.extend(vec![
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.in_proj_z.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.in_proj_b.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.in_proj_a.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.out_proj.weight"),
            strategy: ShardStrategy::RowParallel,
            shard_axis: 1,
            comm_pattern: CommPattern::AllReduce,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.conv1d.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.dt_bias"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.linear_attn.A_log"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 0,
            comm_pattern: CommPattern::None,
        },
    ]);
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "qwen3_5".to_string(),
    })
}

/// Qwen MoE variants.
fn build_qwen_moe_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    // Qwen MoE has a shared expert + routed experts
    plans.extend(moe_expert_plans(prefix, "mlp", config));
    // Shared expert (dense, always present)
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        strategy: ShardStrategy::RowParallel,
        shard_axis: 1,
        comm_pattern: CommPattern::AllReduce,
    });
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "qwen_moe".to_string(),
    })
}

/// Gemma family (Gemma 1/2/3/3n).
fn build_gemma_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "gemma".to_string(),
    })
}

/// Gemma 4 text models: mixed sliding/full attention with optional dense/MoE FFN.
fn build_gemma4_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "language_model.model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    plans.extend(vec![
        LayerShardPlan {
            weight_pattern: format!("{prefix}.experts.switch_glu.gate_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 1,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.experts.switch_glu.up_proj.weight"),
            strategy: ShardStrategy::ColumnParallel,
            shard_axis: 1,
            comm_pattern: CommPattern::None,
        },
        LayerShardPlan {
            weight_pattern: format!("{prefix}.experts.switch_glu.down_proj.weight"),
            strategy: ShardStrategy::RowParallel,
            shard_axis: 2,
            comm_pattern: CommPattern::AllReduce,
        },
    ]);
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "gemma4".to_string(),
    })
}

/// Phi family.
fn build_phi_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    // Phi uses fc1/fc2 naming for FFN in some variants, gate/up/down in others.
    // Include both patterns; the weight loader will match whichever exists.
    plans.extend(ffn_plans(prefix));
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.fc1.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.fc2.weight"),
        strategy: ShardStrategy::RowParallel,
        shard_axis: 1,
        comm_pattern: CommPattern::AllReduce,
    });
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "phi".to_string(),
    })
}

/// Phi MoE.
fn build_phi_moe_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(moe_expert_plans(prefix, "block_sparse_moe", config));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "phimoe".to_string(),
    })
}

/// Mixtral (MoE with block_sparse_moe).
fn build_mixtral_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(moe_expert_plans(prefix, "block_sparse_moe", config));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "mixtral".to_string(),
    })
}

/// OLMoE.
fn build_olmoe_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(moe_expert_plans(prefix, "mlp", config));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "olmoe".to_string(),
    })
}

/// DeepSeek dense (v1).
fn build_deepseek_dense_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "deepseek".to_string(),
    })
}

/// DeepSeek v2/v3 (MLA + MoE). These use latent attention with compressed KV.
fn build_deepseek_moe_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = Vec::new();

    // DeepSeek MLA attention: q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.self_attn.q_a_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.self_attn.q_b_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    // KV projections are replicated (shared compressed KV)
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
        strategy: ShardStrategy::Replicated,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.self_attn.kv_b_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.self_attn.o_proj.weight"),
        strategy: ShardStrategy::RowParallel,
        shard_axis: 1,
        comm_pattern: CommPattern::AllReduce,
    });

    // Dense FFN (first few layers use standard FFN, not MoE)
    plans.extend(ffn_plans(prefix));

    // MoE experts (remaining layers use routed experts)
    plans.extend(moe_expert_plans(prefix, "mlp", config));

    // Shared experts (DeepSeek v2/v3 have shared experts in addition to routed)
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.shared_experts.gate_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.shared_experts.up_proj.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.shared_experts.down_proj.weight"),
        strategy: ShardStrategy::RowParallel,
        shard_axis: 1,
        comm_pattern: CommPattern::AllReduce,
    });

    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "deepseek_moe".to_string(),
    })
}

/// Cohere family (Command R/R+). Uses `model.layers.{}` prefix.
fn build_cohere_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "cohere".to_string(),
    })
}

/// StarCoder2. Uses `model.layers.{}` with standard naming.
fn build_starcoder2_plan(num_layers: usize, config: &ShardConfig) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    // StarCoder2 uses fc1/fc2 naming
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.fc1.weight"),
        strategy: ShardStrategy::ColumnParallel,
        shard_axis: 0,
        comm_pattern: CommPattern::None,
    });
    plans.push(LayerShardPlan {
        weight_pattern: format!("{prefix}.mlp.fc2.weight"),
        strategy: ShardStrategy::RowParallel,
        shard_axis: 1,
        comm_pattern: CommPattern::AllReduce,
    });
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: "starcoder2".to_string(),
    })
}

/// Generic MoE plan (for models using `model.layers.{}` + MoE submodule).
fn build_generic_moe_plan(
    num_layers: usize,
    config: &ShardConfig,
    arch_name: &str,
) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(moe_expert_plans(prefix, "mlp", config));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: arch_name.to_string(),
    })
}

/// Generic transformer plan as a fallback for unrecognized architectures.
fn build_generic_transformer_plan(
    num_layers: usize,
    config: &ShardConfig,
    arch_name: &str,
) -> Result<ModelShardPlan> {
    let prefix = "model.layers.{}";
    let mut plans = attention_plans(prefix);
    plans.extend(ffn_plans(prefix));
    Ok(ModelShardPlan {
        tp_size: config.tp_size,
        num_layers,
        layer_plans: plans,
        embedding_strategy: resolve_embedding_strategy(config.embedding_mode),
        lm_head_strategy: resolve_embedding_strategy(config.lm_head_mode),
        architecture: arch_name.to_string(),
    })
}

#[cfg(test)]
#[path = "plan_generator_tests.rs"]
mod tests;
