//! Single-token Gemma3n decode StableHLO graph.

use super::builder::{Builder, Ty, Val, precision_from_env};
use super::gemma3n::Gemma3nConfig;
use super::gemma3n_emit::input_head;
use super::gemma3n_emit_ops::{
    altup_correct, altup_predict, attention_decode, constants, geglu, gelu, normalize_to, rms_last,
    sparse_gelu,
};
use super::gemma3n_schema::{Input, build_schema, take};

pub(crate) fn emit_gemma3n_decode(c: &Gemma3nConfig, sample: bool) -> String {
    let mut b = Builder::new().with_precision(precision_from_env());
    let (mut decls, args) = build_schema(c, false, 1, true);
    debug_assert!(matches!(args.input, Input::Tokens(_)));
    // `real_len` occupies the generic cache-length position in this schema.
    let mut index = decls.len();
    let cache_ty = Ty::f32(vec![
        c.kv_cache_layers(),
        c.context_capacity,
        c.n_kv,
        c.head_dim,
    ]);
    let mut kcache = take(&mut decls, &mut index, cache_ty.clone(), "kcache");
    let mut vcache = take(&mut decls, &mut index, cache_ty.clone(), "vcache");
    let k = constants(&mut b, c);
    let (base, dense_ple) = input_head(&mut b, c, &k, &args, 1);
    let target = magnitude(&mut b, &base, c.hidden, &k.zero, &k.one);
    let mut planes = vec![base.clone()];
    for projection in &args.weights.initial_projections {
        let projected = b.linear_seq(&base, projection);
        planes.push(normalize_to(&mut b, &projected, &target, &k));
    }
    let cache_map = c.layer_to_cache().expect("validated Gemma3n KV map");
    for (layer, &cache_index) in cache_map.iter().enumerate() {
        let lw = &args.weights.layers[layer];
        let predicted = altup_predict(&mut b, &planes, lw, c, &k, 1);
        let active = &predicted[c.altup_active_idx];
        let normalized = rms_last(&mut b, active, Some(&lw.input_norm), &k.eps, &k.zero);
        let laurel = b.linear_seq(&normalized, &lw.laurel_left);
        let laurel = b.linear_seq(&laurel, &lw.laurel_right);
        let laurel = rms_last(&mut b, &laurel, Some(&lw.laurel_norm), &k.eps, &k.zero);
        let laurel = b.add(&normalized, &laurel);
        let attended = attention_decode(
            &mut b,
            &normalized,
            &args.positions,
            lw,
            layer,
            cache_index,
            c,
            &k,
            &mut kcache,
            &mut vcache,
        );
        let attended = rms_last(&mut b, &attended, Some(&lw.post_attn_norm), &k.eps, &k.zero);
        let sum = b.add(active, &attended);
        let sum = b.add(&sum, &laurel);
        let inv = b.broadcast(&k.inv_sqrt2, &[], vec![1, c.hidden]);
        let residual = b.multiply(&sum, &inv);
        let ff_input = rms_last(&mut b, &residual, Some(&lw.pre_ff_norm), &k.eps, &k.zero);
        let gate = b.linear_seq(&ff_input, &lw.gate);
        let up = b.linear_seq(&ff_input, &lw.up);
        let activated_gate = if c.activation_sparsity[layer] > 0.0 {
            sparse_gelu(&mut b, &gate, c.activation_sparsity[layer], &k)
        } else {
            gelu(&mut b, &gate)
        };
        let mlp = b.multiply(&activated_gate, &up);
        let mlp = b.linear_seq(&mlp, &lw.down);
        let mlp = rms_last(&mut b, &mlp, Some(&lw.post_ff_norm), &k.eps, &k.zero);
        let activated = b.add(&residual, &mlp);
        planes = altup_correct(&mut b, &predicted, &activated, lw, c, &k, 1);
        let mut corrected_active = planes[c.altup_active_idx].clone();
        if c.altup_correct_scale {
            let scale = b.broadcast(&lw.correct_scale, &[1], vec![1, c.hidden]);
            corrected_active = b.multiply(&corrected_active, &scale);
        }
        let ple = b.slice(
            &dense_ple,
            &[(0, 1), (layer, layer + 1), (0, c.hidden_per_layer_input)],
        );
        let ple = b.reshape(&ple, vec![1, c.hidden_per_layer_input]);
        let gate = b.linear_seq(&corrected_active, &lw.ple_gate);
        let injected = geglu(&mut b, &gate, &ple);
        let injected = b.linear_seq(&injected, &lw.ple_projection);
        let injected = rms_last(&mut b, &injected, Some(&lw.ple_norm), &k.eps, &k.zero);
        for (plane, value) in planes.iter_mut().enumerate() {
            if plane != c.altup_active_idx {
                *value = b.add(value, &injected);
            }
        }
    }
    let target = magnitude(
        &mut b,
        &planes[c.altup_active_idx],
        c.hidden,
        &k.zero,
        &k.one,
    );
    let mut collapsed = planes[c.altup_active_idx].clone();
    for (index, projection) in args.weights.unembed_projections.iter().enumerate() {
        let projected = b.linear_seq(&planes[index + 1], projection);
        let projected = normalize_to(&mut b, &projected, &target, &k);
        collapsed = b.add(&collapsed, &projected);
    }
    let count = b.const_f32(c.altup_num_inputs as f32);
    let count = b.broadcast(&count, &[], vec![1, c.hidden]);
    let collapsed = b.divide(&collapsed, &count);
    let normalized = rms_last(
        &mut b,
        &collapsed,
        Some(&args.weights.final_norm),
        &k.eps,
        &k.zero,
    );
    let normalized = b.reshape(&normalized, vec![c.hidden]);
    let logits = b.linear(&normalized, &args.weights.embed);
    let cap = b.const_f32(c.final_logit_softcap);
    let cap = b.broadcast(&cap, &[], vec![c.vocab]);
    let logits = b.divide(&logits, &cap);
    let logits = b.tanh(&logits);
    let logits = b.multiply(&logits, &cap);
    let (result, result_ty) = if sample {
        let token = b.argmax(&logits);
        (token.name, token.ty.render())
    } else {
        (logits.name, logits.ty.render())
    };
    let signature = decls
        .iter()
        .enumerate()
        .map(|(i, d)| format!("%arg{i}: {} loc(\"{}\")", d.ty.render(), d.loc))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "module @decode_step {{\n  func.func public @main({signature}) -> \
         ({result_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {result}, {kc}, {vc} : \
         {result_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        cache_ty = cache_ty.render(),
        body = b.body(),
        kc = kcache.name,
        vc = vcache.name,
    )
}

fn magnitude(b: &mut Builder, value: &Val, hidden: usize, zero: &Val, one: &Val) -> Val {
    let squared = b.multiply(value, value);
    let sum = b.reduce_add(&squared, 1, zero);
    let width = b.const_f32(hidden as f32);
    let width = b.broadcast(&width, &[], vec![1]);
    let mean = b.divide(&sum, &width);
    let inverse = b.rsqrt(&mean);
    let one = b.broadcast(one, &[], vec![1]);
    b.divide(&one, &inverse)
}
