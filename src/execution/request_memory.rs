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

//! Conservative request admission for single-slot Metal Nemotron-H serving.
//!
//! This is an analytical allocation envelope, not a reservation against other
//! GPU clients. Resident *active* allocations include retained prompt snapshots;
//! allocator free-cache is reclaimable and is neither added nor subtracted.

use crate::models::nemotron_h::NemotronHConfig;
use mlxcel_core::memory::MemorySnapshot;

pub const REQUEST_MEMORY_BUDGET_ENV: &str = "MLXCEL_REQUEST_MEMORY_BUDGET";

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "request_memory_budget_exceeded: estimated_bytes={estimated_bytes}, budget_bytes={budget_bytes}; shorten the prompt/output or lower retained cache usage"
)]
pub struct RequestMemoryRejection {
    /// u64::MAX denotes an unknown/overflowing estimate (fail closed).
    pub estimated_bytes: u64,
    /// Zero denotes an unknown runtime budget (fail closed).
    pub budget_bytes: u64,
}

impl RequestMemoryRejection {
    pub const CODE: &'static str = "request_memory_budget_exceeded";
}

/// The shared size grammar, but explicit zero and saturating values are invalid.
/// In particular `none`, NaN, infinity and overflowing sizes cannot disable safety.
pub fn parse_budget(value: &str) -> anyhow::Result<u64> {
    super::runtime::parse_memory_size(value)
        .filter(|bytes| *bytes > 0 && *bytes < u64::MAX)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{REQUEST_MEMORY_BUDGET_ENV} must be a positive finite byte size, for example 12G"
            )
        })
}

pub fn budget_override() -> anyhow::Result<Option<u64>> {
    match std::env::var(REQUEST_MEMORY_BUDGET_ENV) {
        Ok(value) => parse_budget(&value).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(anyhow::anyhow!("{REQUEST_MEMORY_BUDGET_ENV}: {err}")),
    }
}

/// An override may lower a known device budget, never create or raise one.
pub fn resolve_budget(
    recommended: u64,
    wired: u64,
    allocator: u64,
    override_bytes: Option<u64>,
) -> Option<u64> {
    let default = [recommended, wired, allocator]
        .into_iter()
        .filter(|v| *v > 0 && *v < u64::MAX)
        .min()?;
    Some(override_bytes.map_or(default, |value| default.min(value)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestMemoryEstimate {
    pub resident_bytes: u64,
    pub kv_bytes: u64,
    pub recurrent_bytes: u64,
    pub activation_bytes: u64,
    pub ssd_bytes: u64,
    pub total_bytes: u64,
}

fn mul(values: &[u64]) -> Option<u64> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b))
}
fn sum(values: &[u64]) -> Option<u64> {
    values.iter().try_fold(0u64, |a, b| a.checked_add(*b))
}

/// Peak retained positions. With shift enabled the model-owned Nemotron cache
/// does not promise a trim, so do NOT credit the scheduler's dense-pool KV cap.
/// Without shifting the context stop bounds generation; max_tokens=0 still
/// performs one prefill/sample in the existing generation contract.
pub fn retained_tokens(prompt: u64, output: u64, stop_context: Option<u64>) -> Option<u64> {
    let output = output.max(1);
    match stop_context {
        Some(cap) if prompt < cap => Some(prompt.checked_add(output.min(cap - prompt))?),
        Some(_) => None,
        None => prompt.checked_add(output),
    }
}

/// Checked envelope of the *loaded normalized config*. `forward_tokens` is the
/// largest padded piece of the actual cold PrefillPlan (not the global chunk).
/// `element_bytes` is the loaded embedding output dtype (f16/bf16=2, f32=4).
///
/// We deliberately charge cold KV even with a warm snapshot, and triple new KV
/// for growth copies plus snapshot ownership. Layer residuals may stay live in lazy
/// graphs, so they are charged across all layers. Attention scores include a
/// float32 fallback envelope even when fused SDPA normally avoids them. SSD's
/// quadratic scratch uses its *internal* chunk, independent of the outer plan.
pub fn estimate(
    config: &NemotronHConfig,
    forward_tokens: u64,
    retained: u64,
    element_bytes: u64,
    memory: MemorySnapshot,
) -> Option<RequestMemoryEstimate> {
    let c = config;
    let pattern = c.hybrid_override_pattern.as_ref()?;
    if pattern.len() != c.num_hidden_layers
        || pattern.is_empty()
        || forward_tokens == 0
        || retained < forward_tokens
        || ![2, 4].contains(&element_bytes)
        || [
            c.hidden_size,
            c.vocab_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.mamba_num_heads,
            c.mamba_head_dim,
            c.ssm_state_size,
            c.n_groups,
            c.conv_kernel,
            c.chunk_size,
        ]
        .contains(&0)
    {
        return None;
    }
    if pattern
        .iter()
        .any(|p| !matches!(p.as_str(), "*" | "M" | "-" | "E"))
    {
        return None;
    }
    let heads = c.mamba_num_heads as u64;
    let dim = c.mamba_head_dim as u64;
    let state = c.ssm_state_size as u64;
    let hidden = c.hidden_size as u64;
    let attn = pattern.iter().filter(|p| p.as_str() == "*").count() as u64;
    let mamba = pattern.iter().filter(|p| p.as_str() == "M").count() as u64;
    let head_dim = c.head_dim.unwrap_or(c.hidden_size / c.num_attention_heads) as u64;
    if head_dim == 0 {
        return None;
    }
    // Reuse architecture classification, but validate every product before the
    // older startup estimator's saturating arithmetic can hide an overflow.
    let step = mlxcel_core::hardware::KV_CACHE_ALLOC_STEP;
    let rounded = retained
        .checked_add(step - 1)?
        .checked_div(step)?
        .checked_mul(step)?;
    let expected_fp16 = mul(&[attn, 2, c.num_key_value_heads as u64, head_dim, rounded, 2])?;
    let arch = serde_json::json!({
        "model_type": "nemotron_h", "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_key_value_heads,
        "hidden_size": c.hidden_size, "head_dim": head_dim, "hybrid_override_pattern": pattern,
    });
    let classified = super::kv_arch::estimate_kv_arch_from_config(&arch, retained, false, 1)?;
    if classified.total_bytes != expected_fp16 {
        return None;
    }
    let kv_bytes = mul(&[expected_fp16 / 2, element_bytes, 3])?;
    let conv_dim = sum(&[mul(&[heads, dim])?, mul(&[2, c.n_groups as u64, state])?])?;
    let recurrent_bytes = mul(&[
        mamba,
        2,
        sum(&[
            mul(&[heads, dim, state, 4])?,
            mul(&[c.conv_kernel as u64, conv_dim, element_bytes])?,
        ])?,
    ])?;
    let t = forward_tokens;
    let inner = t.min(c.chunk_size as u64);
    // Projection + convolution + float32 x/dtx + grouped B/C + dt chains +
    // accumulated outputs/concatenation. B/C repeats exist only per inner chunk.
    let ssd_bytes = if mamba == 0 {
        0
    } else {
        sum(&[
            mul(&[
                t,
                4,
                sum(&[
                    mul(&[12, heads, dim])?,
                    mul(&[6, c.n_groups as u64, state])?,
                    mul(&[8, heads])?,
                ])?,
            ])?,
            mul(&[inner, inner, heads, 4, 10])?,
            mul(&[inner, heads, state, 4, 4])?,
            mul(&[heads, dim, state, 4, 4])?,
        ])?
    };
    let experts = c.num_experts_per_tok.unwrap_or(1).max(1) as u64;
    let intermediate =
        (c.intermediate_size as u64).max(c.moe_intermediate_size.unwrap_or(0) as u64);
    let activation_bytes = sum(&[
        mul(&[t, hidden, c.num_hidden_layers as u64, element_bytes, 4])?,
        mul(&[t, c.vocab_size as u64, 4, 2])?,
        mul(&[t, intermediate, experts, 4, 4])?,
        mul(&[
            t,
            c.moe_shared_expert_intermediate_size.unwrap_or(0) as u64,
            4,
            4,
        ])?,
        // Matmul temporary/dequantization workspace, counted even if the
        // quantized kernel avoids it; no second copy of resident model weights.
        mul(&[hidden, intermediate.max(conv_dim), 4, 3, experts])?,
        if attn == 0 {
            0
        } else {
            mul(&[t, retained, c.num_attention_heads as u64, 4, 3])?
        },
    ])?;
    let total_bytes = sum(&[
        memory.active_bytes,
        kv_bytes,
        recurrent_bytes,
        activation_bytes,
        ssd_bytes,
    ])?;
    Some(RequestMemoryEstimate {
        resident_bytes: memory.active_bytes,
        kv_bytes,
        recurrent_bytes,
        activation_bytes,
        ssd_bytes,
        total_bytes,
    })
}

pub fn check(
    estimate: Option<RequestMemoryEstimate>,
    budget: Option<u64>,
) -> Result<RequestMemoryEstimate, RequestMemoryRejection> {
    match (estimate, budget) {
        (Some(estimate), Some(budget)) if budget > 0 && estimate.total_bytes <= budget => {
            Ok(estimate)
        }
        (estimate, budget) => Err(RequestMemoryRejection {
            estimated_bytes: estimate.map_or(u64::MAX, |e| e.total_bytes),
            budget_bytes: budget.unwrap_or(0),
        }),
    }
}

#[cfg(test)]
#[path = "request_memory_tests.rs"]
mod tests;
