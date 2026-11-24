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

//! Issue #3362: the Phi4MM LoRA A/B arguments the emitter declares must match
//! the shapes the loader uploads, in both projection layouts and every graph
//! kind. The single-sequence `InOut` layout (issue #2859) transposes `Proj`
//! specs at upload, so a LoRA pair pushed as `Proj` arrived as `[in, rank]`
//! against a declared `[rank, in]` (the `320x3072` vs `3072x320` prefill error).

use std::collections::HashMap;

use super::{WeightSpec, weight_specs};
use crate::emitter::{
    Config, Phi4LoraConfig, ProjectionLayout, emit_decode, emit_decode_ragged, emit_prefill,
    emit_prefill_embeddings,
};

fn phi4_config(layout: ProjectionLayout) -> Config {
    let mut c = Config {
        context_capacity: 8,
        hidden: 8,
        inter: 16,
        n_layers: 2,
        n_q: 2,
        n_kv: 1,
        head_dim: 4,
        rotary_dim: Some(4),
        vocab: 32,
        tie_word_embeddings: true,
        fused_qkv: true,
        fused_gate_up: true,
        phi4_lora: Some(Phi4LoraConfig {
            speech_rank: 3,
            speech_scale: 2.0,
            vision_rank: 2,
            vision_scale: 2.5,
        }),
        ..Config::llama_3_2_1b()
    };
    c.projection_layout = layout;
    c
}

/// Map a LoRA tensor name to the emitter's argument location and the
/// checkpoint shape (PEFT stores `lora_A.weight` as `[rank, in]` and
/// `lora_B.weight` as `[out, rank]`).
fn lora_arg(c: &Config, name: &str) -> Option<(String, Vec<usize>)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_suffix(".weight")?;
    let (module, adapter_part) = rest.split_once(".lora_")?;
    let (matrix, adapter) = adapter_part.split_once('.')?;
    let lora = c.phi4_lora?;
    let rank = match adapter {
        "speech" => lora.speech_rank,
        "vision" => lora.vision_rank,
        other => panic!("unknown Phi4MM adapter {other}"),
    };
    let (h, qd, kv, inter) = (c.hidden, c.n_q * c.head_dim, c.n_kv * c.head_dim, c.inter);
    let (stem, input, output) = match module {
        "self_attn.qkv_proj" => ("qkv", h, qd + 2 * kv),
        "self_attn.o_proj" => ("o", qd, h),
        "mlp.gate_up_proj" => ("gate_up", h, 2 * inter),
        "mlp.down_proj" => ("down", inter, h),
        other => panic!("unknown Phi4MM LoRA module {other}"),
    };
    let shape = match matrix {
        "A" => vec![rank, input],
        "B" => vec![output, rank],
        other => panic!("unknown LoRA matrix {other}"),
    };
    let loc = format!("params['layers'][{layer}]['{stem}.lora_{matrix}.{adapter}']");
    Some((loc, shape))
}

/// The declared `tensor<AxBxf32>` shape and element type of every argument,
/// keyed by its `loc`.
fn declared_args(graph: &str) -> HashMap<String, (Vec<usize>, String)> {
    let mut args = HashMap::new();
    for segment in graph.split("%arg").skip(1) {
        let Some(ty_start) = segment.find(": tensor<") else {
            continue;
        };
        let ty = &segment[ty_start + ": tensor<".len()..];
        let Some(ty_end) = ty.find('>') else {
            continue;
        };
        let ty = &ty[..ty_end];
        let after = &segment[ty_start..];
        let Some(loc_start) = after.find("loc(\"") else {
            continue;
        };
        let loc = &after[loc_start + "loc(\"".len()..];
        let Some(loc_end) = loc.find("\")") else {
            continue;
        };
        let mut parts: Vec<&str> = ty.split('x').collect();
        let elt = parts.pop().unwrap_or_default().to_string();
        let shape = parts
            .iter()
            .map(|d| d.parse().expect("static dim"))
            .collect();
        args.insert(loc[..loc_end].to_string(), (shape, elt));
    }
    args
}

#[test]
fn phi4mm_lora_declared_shapes_match_uploaded_shapes_in_every_layout() {
    for layout in [ProjectionLayout::InOut, ProjectionLayout::OutIn] {
        let c = phi4_config(layout);
        let specs = weight_specs(&c);
        let lora_specs: Vec<&WeightSpec> = specs
            .iter()
            .filter(|spec| spec.tensor_name().contains(".lora_"))
            .collect();
        assert_eq!(lora_specs.len(), c.n_layers * 4 * 2 * 2);

        let graphs = [
            ("prefill", emit_prefill(&c, false)),
            (
                "prefill_embeddings",
                emit_prefill_embeddings(&c, false).expect("embeddings prefill"),
            ),
            ("decode", emit_decode(&c, false)),
            ("ragged", emit_decode_ragged(&c, 4, false).expect("ragged")),
        ];
        for (kind, graph) in &graphs {
            let declared = declared_args(graph);
            for spec in &lora_specs {
                let (loc, checkpoint) = lora_arg(&c, spec.tensor_name()).expect("LoRA name");
                let uploaded = if spec.uploads_transposed(layout) {
                    checkpoint.iter().rev().copied().collect()
                } else {
                    checkpoint.clone()
                };
                let (shape, elt) = declared
                    .get(&loc)
                    .unwrap_or_else(|| panic!("{kind} ({layout:?}) does not declare {loc}"));
                assert_eq!(
                    shape, &uploaded,
                    "{kind} ({layout:?}) declares {loc} as {shape:?} but the loader uploads {uploaded:?}"
                );
                assert_eq!(
                    shape, &checkpoint,
                    "{kind} ({layout:?}) must keep {loc} in the checkpoint layout"
                );
                assert_eq!(elt, "f32", "{kind} ({layout:?}) {loc} is f32-resident");
                assert!(
                    matches!(spec, WeightSpec::Whole(_)),
                    "{} must not be a transposable projection spec",
                    spec.tensor_name()
                );
            }
        }
    }
}

#[test]
fn only_projection_specs_upload_transposed() {
    let proj = WeightSpec::Proj("w".to_string());
    let rows = WeightSpec::Rows {
        name: "w".to_string(),
        start: 0,
        end: 1,
    };
    let whole = WeightSpec::Whole("w".to_string());
    assert!(proj.uploads_transposed(ProjectionLayout::InOut));
    assert!(rows.uploads_transposed(ProjectionLayout::InOut));
    assert!(!whole.uploads_transposed(ProjectionLayout::InOut));
    for spec in [&proj, &rows, &whole] {
        assert!(!spec.uploads_transposed(ProjectionLayout::OutIn));
    }
}
