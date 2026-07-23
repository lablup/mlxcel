//! Gemma3n StableHLO prefill graphs.
//!
//! Both entries share the complete AltUp/LAUREL/shared-KV backbone. Token
//! prefill gathers ordinary and per-layer embeddings; multimodal prefill accepts
//! already post-scale merged embeddings plus dense projected PLE.

use super::builder::{Builder, Precision, Val, precision_from_env};
use super::gemma3n::{Gemma3nConfig, Gemma3nLayerType};
use super::gemma3n_emit_ops::{
    Constants, altup_correct, altup_predict, apply_sliding_window, attention, causal_mask,
    constants, geglu, normalize_to, rms_last, sparse_gelu,
};
use super::gemma3n_schema::{Args, Decl, Input, build_schema};

pub(crate) fn emit_gemma3n_prefill(c: &Gemma3nConfig, sample: bool) -> String {
    emit(c, sample, false, precision_from_env())
}

pub(crate) fn emit_gemma3n_prefill_embeddings_ple(c: &Gemma3nConfig, sample: bool) -> String {
    emit(c, sample, true, precision_from_env())
}

fn emit(c: &Gemma3nConfig, sample: bool, prepared: bool, precision: Precision) -> String {
    let lp = c.context_capacity;
    let mut b = Builder::new().with_precision(precision);
    let (decls, args) = build_schema(c, prepared, lp, false);
    let k = constants(&mut b, c);
    let (base, dense_ple) = input_head(&mut b, c, &k, &args, lp);
    let target = magnitude(&mut b, &base, c, &k);
    let mut planes = vec![base.clone()];
    for projection in &args.weights.initial_projections {
        let projected = b.linear_seq(&base, projection);
        planes.push(normalize_to(&mut b, &projected, &target, &k));
    }
    let full_mask = args
        .attention_bias
        .clone()
        .unwrap_or_else(|| causal_mask(&mut b, lp, None, &k));
    let sliding_mask = apply_sliding_window(&mut b, &full_mask, lp, c.sliding_window, &k);
    let cache_map = c.layer_to_cache().expect("validated Gemma3n KV map");
    let cache_shape = vec![c.kv_cache_layers(), lp, c.n_kv, c.head_dim];
    let mut kcache = b.broadcast(&k.zero, &[], cache_shape.clone());
    let mut vcache = b.broadcast(&k.zero, &[], cache_shape.clone());
    for (layer, &cache_index) in cache_map.iter().enumerate() {
        let lw = &args.weights.layers[layer];
        let predicted = altup_predict(&mut b, &planes, lw, c, &k, lp);
        let active = &predicted[c.altup_active_idx];
        let normalized = rms_last(&mut b, active, Some(&lw.input_norm), &k.eps, &k.zero);
        let laurel = b.linear_seq(&normalized, &lw.laurel_left);
        let laurel = b.linear_seq(&laurel, &lw.laurel_right);
        let laurel = rms_last(&mut b, &laurel, Some(&lw.laurel_norm), &k.eps, &k.zero);
        let laurel = b.add(&normalized, &laurel);
        let mask = match c.layer_types[layer] {
            Gemma3nLayerType::Full => &full_mask,
            Gemma3nLayerType::Sliding => &sliding_mask,
        };
        let attended = attention(
            &mut b,
            &normalized,
            &args.positions,
            mask,
            lw,
            layer,
            cache_index,
            c,
            &k,
            &mut kcache,
            &mut vcache,
        );
        let attended = rms_last(&mut b, &attended, Some(&lw.post_attn_norm), &k.eps, &k.zero);
        let inv = b.broadcast(&k.inv_sqrt2, &[], vec![lp, c.hidden]);
        let residual = b.add(active, &attended);
        let residual = b.add(&residual, &laurel);
        let residual = b.multiply(&residual, &inv);
        let ff_input = rms_last(&mut b, &residual, Some(&lw.pre_ff_norm), &k.eps, &k.zero);
        let gate = b.linear_seq(&ff_input, &lw.gate);
        let up = b.linear_seq(&ff_input, &lw.up);
        let activated_gate = if c.activation_sparsity[layer] > 0.0 {
            sparse_gelu(&mut b, &gate, c.activation_sparsity[layer], &k)
        } else {
            super::gemma3n_emit_ops::gelu(&mut b, &gate)
        };
        let mlp = b.multiply(&activated_gate, &up);
        let mlp = b.linear_seq(&mlp, &lw.down);
        let mlp = rms_last(&mut b, &mlp, Some(&lw.post_ff_norm), &k.eps, &k.zero);
        let activated = b.add(&residual, &mlp);
        planes = altup_correct(&mut b, &predicted, &activated, lw, c, &k, lp);
        let mut corrected_active = planes[c.altup_active_idx].clone();
        if c.altup_correct_scale {
            let scale = b.broadcast(&lw.correct_scale, &[1], vec![lp, c.hidden]);
            corrected_active = b.multiply(&corrected_active, &scale);
        }
        let ple = b.slice(
            &dense_ple,
            &[(0, lp), (layer, layer + 1), (0, c.hidden_per_layer_input)],
        );
        let ple = b.reshape(&ple, vec![lp, c.hidden_per_layer_input]);
        let ple_gate = b.linear_seq(&corrected_active, &lw.ple_gate);
        let injected = geglu(&mut b, &ple_gate, &ple);
        let injected = b.linear_seq(&injected, &lw.ple_projection);
        let injected = rms_last(&mut b, &injected, Some(&lw.ple_norm), &k.eps, &k.zero);
        for (plane, value) in planes.iter_mut().enumerate() {
            if plane != c.altup_active_idx {
                *value = b.add(value, &injected);
            }
        }
    }
    let target = magnitude(&mut b, &planes[c.altup_active_idx], c, &k);
    let mut collapsed = planes[c.altup_active_idx].clone();
    for (index, projection) in args.weights.unembed_projections.iter().enumerate() {
        let projected = b.linear_seq(&planes[index + 1], projection);
        let projected = normalize_to(&mut b, &projected, &target, &k);
        collapsed = b.add(&collapsed, &projected);
    }
    let count = b.const_f32(c.altup_num_inputs as f32);
    let count = b.broadcast(&count, &[], vec![lp, c.hidden]);
    collapsed = b.divide(&collapsed, &count);
    let normalized = rms_last(
        &mut b,
        &collapsed,
        Some(&args.weights.final_norm),
        &k.eps,
        &k.zero,
    );
    let one = b.const_i32(1);
    let last_index = b.subtract(&args.real_len, &one);
    let zero = b.const_i32(0);
    let row = b.dynamic_slice(&normalized, &[&last_index, &zero], vec![1, c.hidden]);
    let row = b.reshape(&row, vec![c.hidden]);
    let logits = b.linear(&row, &args.weights.embed);
    let cap = b.const_f32(c.final_logit_softcap);
    let cap_b = b.broadcast(&cap, &[], vec![c.vocab]);
    let logits = b.divide(&logits, &cap_b);
    let logits = b.tanh(&logits);
    let logits = b.multiply(&logits, &cap_b);
    let (result, result_ty) = if sample {
        let token = b.argmax(&logits);
        (token.name, token.ty.render())
    } else {
        (logits.name, logits.ty.render())
    };
    render_module(
        if prepared {
            "prefill_embeddings_ple"
        } else {
            "prefill"
        },
        &decls,
        &b,
        &result,
        &result_ty,
        &kcache,
        &vcache,
    )
}

pub(super) fn input_head(
    b: &mut Builder,
    c: &Gemma3nConfig,
    k: &Constants,
    a: &Args,
    lp: usize,
) -> (Val, Val) {
    let ple_width = c.n_layers * c.hidden_per_layer_input;
    match &a.input {
        Input::Prepared {
            embeddings,
            dense_ple,
        } => (embeddings.clone(), dense_ple.clone()),
        Input::Tokens(tokens) => {
            let tokens = if tokens.ty.shape.is_empty() {
                b.reshape(tokens, vec![1])
            } else {
                tokens.clone()
            };
            let indices = b.reshape(&tokens, vec![lp, 1]);
            let base = b.gather(&a.weights.embed, &indices);
            let scale = b.const_f32((c.hidden as f32).sqrt());
            let scale = b.broadcast(&scale, &[], vec![lp, c.hidden]);
            let base = b.multiply(&base, &scale);
            let limit = b.const_i32(c.per_layer_vocab as i32);
            let limit = b.broadcast(&limit, &[], vec![lp]);
            let zero_i = b.const_i32(0);
            let zero_ids = b.broadcast(&zero_i, &[], vec![lp]);
            let nonnegative = b.compare("GE", &tokens, &zero_ids, "SIGNED");
            let below = b.compare("LT", &tokens, &limit, "SIGNED");
            let valid = b.select(&nonnegative, &below, &nonnegative);
            let safe = b.select(&valid, &tokens, &zero_ids);
            let safe = b.reshape(&safe, vec![lp, 1]);
            let token_ple = b.gather(&a.weights.token_ple, &safe);
            let zeros = b.broadcast(&k.zero, &[], vec![lp, ple_width]);
            let valid = b.broadcast(&valid, &[0], vec![lp, ple_width]);
            let token_ple = b.select(&valid, &token_ple, &zeros);
            let token_scale = b.const_f32((c.hidden_per_layer_input as f32).sqrt());
            let token_scale = b.broadcast(&token_scale, &[], vec![lp, ple_width]);
            let token_ple = b.multiply(&token_ple, &token_scale);
            let projected = b.linear_seq(&base, &a.weights.ple_projection);
            let model_scale = b.const_f32((c.hidden as f32).sqrt().recip());
            let model_scale = b.broadcast(&model_scale, &[], vec![lp, ple_width]);
            let projected = b.multiply(&projected, &model_scale);
            let projected = b.reshape(&projected, vec![lp, c.n_layers, c.hidden_per_layer_input]);
            let projected = rms_last(
                b,
                &projected,
                Some(&a.weights.ple_projection_norm),
                &k.eps,
                &k.zero,
            );
            let token_ple = b.reshape(&token_ple, vec![lp, c.n_layers, c.hidden_per_layer_input]);
            let inv = b.broadcast(
                &k.inv_sqrt2,
                &[],
                vec![lp, c.n_layers, c.hidden_per_layer_input],
            );
            let combined = b.add(&projected, &token_ple);
            let combined = b.multiply(&combined, &inv);
            (base, combined)
        }
    }
}

fn magnitude(b: &mut Builder, value: &Val, c: &Gemma3nConfig, k: &Constants) -> Val {
    let sq = b.multiply(value, value);
    let sum = b.reduce_add(&sq, 1, &k.zero);
    let width = b.broadcast(&k.hidden, &[], vec![c.context_capacity]);
    let mean = b.divide(&sum, &width);
    let inverse = b.rsqrt(&mean);
    let one = b.broadcast(&k.one, &[], vec![c.context_capacity]);
    b.divide(&one, &inverse)
}

fn render_module(
    name: &str,
    decls: &[Decl],
    b: &Builder,
    result: &str,
    result_ty: &str,
    kcache: &Val,
    vcache: &Val,
) -> String {
    let signature = decls
        .iter()
        .enumerate()
        .map(|(index, decl)| format!("%arg{index}: {} loc(\"{}\")", decl.ty.render(), decl.loc))
        .collect::<Vec<_>>()
        .join(", ");
    let cache_ty = kcache.ty.render();
    format!(
        "module @{name} {{\n  func.func public @main({signature}) -> ({result_ty}, \
         {cache_ty}, {cache_ty}) {{\n{body}    return {result}, {kc}, {vc} : \
         {result_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        body = b.body(),
        kc = kcache.name,
        vc = vcache.name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Gemma3nConfig {
        Gemma3nConfig::from_json_str(
            &serde_json::json!({
                "model_type": "gemma3n_text",
                "hidden_size": 8, "intermediate_size": [12, 12, 12, 12],
                "num_hidden_layers": 4, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6,
                "vocab_size": 12, "vocab_size_per_layer_input": 10,
                "hidden_size_per_layer_input": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "activation_sparsity_pattern": [0.5, 0.0, 0.0, 0.0],
                "sliding_window": 2, "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0, "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 2, "altup_num_inputs": 2,
                "altup_active_idx": 0, "altup_coef_clip": 120.0,
                "altup_correct_scale": true, "laurel_rank": 2,
                "tie_word_embeddings": true
            })
            .to_string(),
        )
        .unwrap()
        .with_context_capacity(4)
        .unwrap()
    }

    #[test]
    fn emits_distinct_token_and_dense_ple_entries_with_physical_kv_count() {
        let c = tiny();
        let token = emit_gemma3n_prefill(&c, false);
        let prepared = emit_gemma3n_prefill_embeddings_ple(&c, false);
        assert!(token.contains("module @prefill"));
        assert!(!token.contains("loc(\"dense_ple\")"));
        assert!(prepared.contains("module @prefill_embeddings_ple"));
        assert!(prepared.contains("tensor<4x4x2xf32> loc(\"dense_ple\")"));
        assert!(prepared.contains("tensor<2x4x1x4xf32>"));
        assert!(prepared.contains("altup.prediction_coefs.weight"));
        assert!(prepared.contains("laurel.linear_left.weight"));
        assert!(!prepared.contains("layers.2.self_attn.k_proj.weight"));
        let decode = super::super::gemma3n_decode::emit_gemma3n_decode(&c, true);
        assert!(decode.contains("module @decode_step"));
        assert!(decode.contains("tensor<2x4x1x4xf32> loc(\"kcache\")"));
        assert!(!decode.contains("layers.2.self_attn.k_proj.weight"));
    }

    #[test]
    #[ignore = "requires the pinned IREE compiler; run explicitly for the production target"]
    fn iree_compiles_tiny_token_and_dense_ple_graphs() {
        let compiler = std::env::var_os("MLXCEL_XLA_IREE_COMPILE")
            .expect("set MLXCEL_XLA_IREE_COMPILE to the pinned iree-compile");
        let target =
            std::env::var("MLXCEL_XLA_IREE_TEST_TARGET").unwrap_or_else(|_| "local".to_string());
        for (tag, graph) in [
            ("token", emit_gemma3n_prefill(&tiny(), true)),
            (
                "embeddings-ple",
                emit_gemma3n_prefill_embeddings_ple(&tiny(), true),
            ),
            (
                "decode",
                super::super::gemma3n_decode::emit_gemma3n_decode(&tiny(), true),
            ),
        ] {
            let stem = format!("mlxcel-gemma3n-{tag}-{}", std::process::id());
            let input = std::env::temp_dir().join(format!("{stem}.mlir"));
            let output = std::env::temp_dir().join(format!("{stem}.vmfb"));
            std::fs::write(&input, graph).unwrap();
            let mut command = std::process::Command::new(&compiler);
            command.arg("--iree-input-type=stablehlo");
            if target == "cuda" {
                command.arg("--iree-hal-target-device=cuda");
            } else {
                command
                    .arg("--iree-hal-target-device=local")
                    .arg("--iree-hal-local-target-device-backends=llvm-cpu");
            }
            let result = command.arg(&input).arg("-o").arg(&output).output().unwrap();
            let _ = std::fs::remove_file(&input);
            let _ = std::fs::remove_file(&output);
            assert!(
                result.status.success(),
                "{tag} failed IREE compile: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
