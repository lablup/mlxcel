//! Stable Gemma3n graph argument schema shared by prefill and decode.

use super::builder::{Builder, Ty, Val};
use super::gemma3n::Gemma3nConfig;
use super::gemma3n_emit_ops::LayerWeights;

pub(super) struct Decl {
    pub ty: Ty,
    pub loc: String,
}

pub(super) struct Weights {
    pub embed: Val,
    pub token_ple: Val,
    pub ple_projection: Val,
    pub ple_projection_norm: Val,
    pub final_norm: Val,
    pub initial_projections: Vec<Val>,
    pub unembed_projections: Vec<Val>,
    pub layers: Vec<LayerWeights>,
}

pub(super) enum Input {
    Tokens(Val),
    Prepared { embeddings: Val, dense_ple: Val },
}

pub(super) struct Args {
    pub weights: Weights,
    pub input: Input,
    pub positions: Val,
    pub real_len: Val,
    pub attention_bias: Option<Val>,
}

pub(super) fn take(
    decls: &mut Vec<Decl>,
    index: &mut usize,
    ty: Ty,
    loc: impl Into<String>,
) -> Val {
    let value = Builder::arg(*index, ty.clone());
    decls.push(Decl {
        ty,
        loc: loc.into(),
    });
    *index += 1;
    value
}

pub(super) fn build_schema(
    c: &Gemma3nConfig,
    prepared: bool,
    rows: usize,
    scalar_inputs: bool,
    vector_real_len: bool,
) -> (Vec<Decl>, Args) {
    let mut decls = Vec::new();
    let mut index = 0;
    let root = "model.language_model";
    let h = c.hidden;
    let ple_width = c.n_layers * c.hidden_per_layer_input;
    let projection =
        |decls: &mut Vec<Decl>, index: &mut usize, out: usize, in_: usize, name: String| {
            take(decls, index, Ty::f32(vec![out, in_]), name)
        };
    let embed = projection(
        &mut decls,
        &mut index,
        c.vocab,
        h,
        format!("{root}.embed_tokens.weight"),
    );
    let token_ple = projection(
        &mut decls,
        &mut index,
        c.per_layer_vocab,
        ple_width,
        format!("{root}.embed_tokens_per_layer.weight"),
    );
    let ple_projection = projection(
        &mut decls,
        &mut index,
        ple_width,
        h,
        format!("{root}.per_layer_model_projection.weight"),
    );
    let ple_projection_norm = take(
        &mut decls,
        &mut index,
        Ty::f32(vec![c.hidden_per_layer_input]),
        format!("{root}.per_layer_projection_norm.weight"),
    );
    let final_norm = take(
        &mut decls,
        &mut index,
        Ty::f32(vec![h]),
        format!("{root}.norm.weight"),
    );
    let initial_projections = (0..c.altup_num_inputs - 1)
        .map(|plane| {
            projection(
                &mut decls,
                &mut index,
                h,
                h,
                format!("{root}.altup_projections.{plane}.weight"),
            )
        })
        .collect();
    let unembed_projections = (0..c.altup_num_inputs - 1)
        .map(|plane| {
            projection(
                &mut decls,
                &mut index,
                h,
                h,
                format!("{root}.altup_unembed_projections.{plane}.weight"),
            )
        })
        .collect();
    let mut layers = Vec::with_capacity(c.n_layers);
    for layer in 0..c.n_layers {
        let p = format!("{root}.layers.{layer}");
        let concrete = layer < c.kv_cache_layers();
        layers.push(LayerWeights {
            correct_scale: take(
                &mut decls,
                &mut index,
                Ty::f32(vec![h]),
                format!("{p}.altup.correct_output_scale"),
            ),
            correction: take(
                &mut decls,
                &mut index,
                Ty::f32(vec![c.altup_num_inputs, c.altup_num_inputs]),
                format!("{p}.altup.correction_coefs.weight"),
            ),
            router: projection(
                &mut decls,
                &mut index,
                c.altup_num_inputs,
                h,
                format!("{p}.altup.modality_router.weight"),
            ),
            router_norm: take(
                &mut decls,
                &mut index,
                Ty::f32(vec![h]),
                format!("{p}.altup.router_norm.weight"),
            ),
            prediction: take(
                &mut decls,
                &mut index,
                Ty::f32(vec![
                    c.altup_num_inputs * c.altup_num_inputs,
                    c.altup_num_inputs,
                ]),
                format!("{p}.altup.prediction_coefs.weight"),
            ),
            laurel_left: projection(
                &mut decls,
                &mut index,
                c.laurel_rank,
                h,
                format!("{p}.laurel.linear_left.weight"),
            ),
            laurel_right: projection(
                &mut decls,
                &mut index,
                h,
                c.laurel_rank,
                format!("{p}.laurel.linear_right.weight"),
            ),
            laurel_norm: take(
                &mut decls,
                &mut index,
                Ty::f32(vec![h]),
                format!("{p}.laurel.post_laurel_norm.weight"),
            ),
            input_norm: norm(
                &mut decls,
                &mut index,
                h,
                format!("{p}.input_layernorm.weight"),
            ),
            post_attn_norm: norm(
                &mut decls,
                &mut index,
                h,
                format!("{p}.post_attention_layernorm.weight"),
            ),
            pre_ff_norm: norm(
                &mut decls,
                &mut index,
                h,
                format!("{p}.pre_feedforward_layernorm.weight"),
            ),
            post_ff_norm: norm(
                &mut decls,
                &mut index,
                h,
                format!("{p}.post_feedforward_layernorm.weight"),
            ),
            wq: projection(
                &mut decls,
                &mut index,
                c.n_q * c.head_dim,
                h,
                format!("{p}.self_attn.q_proj.weight"),
            ),
            wk: concrete.then(|| {
                projection(
                    &mut decls,
                    &mut index,
                    c.n_kv * c.head_dim,
                    h,
                    format!("{p}.self_attn.k_proj.weight"),
                )
            }),
            wv: concrete.then(|| {
                projection(
                    &mut decls,
                    &mut index,
                    c.n_kv * c.head_dim,
                    h,
                    format!("{p}.self_attn.v_proj.weight"),
                )
            }),
            wo: projection(
                &mut decls,
                &mut index,
                h,
                c.n_q * c.head_dim,
                format!("{p}.self_attn.o_proj.weight"),
            ),
            q_norm: norm(
                &mut decls,
                &mut index,
                c.head_dim,
                format!("{p}.self_attn.q_norm.weight"),
            ),
            k_norm: concrete.then(|| {
                norm(
                    &mut decls,
                    &mut index,
                    c.head_dim,
                    format!("{p}.self_attn.k_norm.weight"),
                )
            }),
            gate: projection(
                &mut decls,
                &mut index,
                c.intermediate[layer],
                h,
                format!("{p}.mlp.gate_proj.weight"),
            ),
            up: projection(
                &mut decls,
                &mut index,
                c.intermediate[layer],
                h,
                format!("{p}.mlp.up_proj.weight"),
            ),
            down: projection(
                &mut decls,
                &mut index,
                h,
                c.intermediate[layer],
                format!("{p}.mlp.down_proj.weight"),
            ),
            ple_gate: projection(
                &mut decls,
                &mut index,
                c.hidden_per_layer_input,
                h,
                format!("{p}.per_layer_input_gate.weight"),
            ),
            ple_projection: projection(
                &mut decls,
                &mut index,
                h,
                c.hidden_per_layer_input,
                format!("{p}.per_layer_projection.weight"),
            ),
            ple_norm: norm(
                &mut decls,
                &mut index,
                h,
                format!("{p}.post_per_layer_input_norm.weight"),
            ),
        });
    }
    let input = if prepared {
        Input::Prepared {
            embeddings: take(&mut decls, &mut index, Ty::f32(vec![rows, h]), "embeddings"),
            dense_ple: take(
                &mut decls,
                &mut index,
                Ty::f32(vec![rows, c.n_layers, c.hidden_per_layer_input]),
                "dense_ple",
            ),
        }
    } else {
        Input::Tokens(take(
            &mut decls,
            &mut index,
            if scalar_inputs {
                Ty::scalar("i32")
            } else {
                Ty::new(vec![rows], "i32")
            },
            "tokens",
        ))
    };
    let positions = take(
        &mut decls,
        &mut index,
        if scalar_inputs {
            Ty::scalar("i32")
        } else {
            Ty::new(vec![rows], "i32")
        },
        "positions",
    );
    let real_len = take(
        &mut decls,
        &mut index,
        if vector_real_len {
            Ty::new(vec![rows], "i32")
        } else {
            Ty::scalar("i32")
        },
        "real_len",
    );
    let attention_bias = prepared.then(|| {
        take(
            &mut decls,
            &mut index,
            Ty::f32(vec![rows, rows]),
            "attention_bias",
        )
    });
    (
        decls,
        Args {
            weights: Weights {
                embed,
                token_ple,
                ple_projection,
                ple_projection_norm,
                final_norm,
                initial_projections,
                unembed_projections,
                layers,
            },
            input,
            positions,
            real_len,
            attention_bias,
        },
    )
}

fn norm(decls: &mut Vec<Decl>, index: &mut usize, width: usize, name: String) -> Val {
    take(decls, index, Ty::f32(vec![width]), name)
}
