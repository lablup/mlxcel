// Measurement probe for lablup/mlxcel#2154: granite-4.0-h-tiny single-token
// (`w1`) forwards with an f32-activation reference, per-op bf16 error against
// f32 on the same inputs, and mixed-precision arms that run one op family in
// bf16 inside an otherwise f32 forward. Not built by default; README.md in this
// directory says how to attach and run it.

use super::*;
use mlxcel_core::{astype, dtype, eval, from_slice_i32};
use std::collections::BTreeMap;
use std::io::Write;

fn f32v(a: &MlxArray) -> Vec<f32> {
    let a = astype(a, dtype::FLOAT32);
    eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn set_gpu(gpu: bool) {
    mlxcel_core::set_default_device(gpu);
}

#[derive(Default, Clone)]
struct Acc {
    n: usize,
    rel: f64,
    rel_max: f64,
    abs_max: f64,
}

impl Acc {
    fn add(&mut self, g: &[f32], c: &[f32]) {
        let mut num = 0f64;
        let mut den = 0f64;
        let mut am = 0f64;
        for (x, y) in g.iter().zip(c) {
            let d = (*x as f64) - (*y as f64);
            num += d * d;
            den += (*y as f64) * (*y as f64);
            am = am.max(d.abs());
        }
        let r = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
        self.n += 1;
        self.rel += r;
        self.rel_max = self.rel_max.max(r);
        self.abs_max = self.abs_max.max(am);
    }
}

type Ctx = BTreeMap<String, UniquePtr<MlxArray>>;

fn put(ctx: &mut Ctx, k: &str, a: UniquePtr<MlxArray>) {
    eval(&a);
    ctx.insert(k.to_string(), a);
}

/// One named stage: computes its output from the CPU context only.
type Stage<'a> = (&'static str, Box<dyn Fn(&Ctx) -> UniquePtr<MlxArray> + 'a>);

fn mamba_stages<'a>(layer: &'a GraniteMoeHybridDecoderLayer) -> Vec<Stage<'a>> {
    let Mixer::Mamba(m) = &layer.mixer else {
        unreachable!()
    };
    let isz = m.intermediate_size as i32;
    let cdim = m.conv_dim as i32;
    let bc = (m.n_groups * m.ssm_state_size) as i32;
    vec![
        (
            "norm1",
            Box::new(move |c: &Ctx| layer.input_layernorm.forward(&c["x"])),
        ),
        (
            "m.in_proj",
            Box::new(move |c: &Ctx| m.in_proj.forward(&c["norm1"])),
        ),
        (
            "m.conv_silu",
            Box::new(move |c: &Ctx| {
                let p = &c["m.in_proj"];
                let ci = slice_axis(p, -1, isz, isz + cdim);
                let dt = mlxcel_core::array_dtype(&ci);
                let pad = mlxcel_core::zeros(&[1, (m.conv_kernel_size - 1) as i32, cdim], dt);
                let padded = concatenate(&pad, &ci, 1);
                let o = mlxcel_core::conv1d(&padded, &m.conv_weight, 1, 0, 1, cdim);
                let o = if let Some(ref b) = m.conv_bias {
                    mlxcel_core::add(&o, &mlxcel_core::reshape(b, &[1, 1, -1]))
                } else {
                    o
                };
                silu(&o)
            }),
        ),
        (
            "m.ssm_step",
            Box::new(move |c: &Ctx| {
                let p = &c["m.in_proj"];
                let co = &c["m.conv_silu"];
                let dt = slice_axis(p, -1, isz + cdim, -1);
                let hs = slice_axis(co, -1, 0, isz);
                let b = slice_axis(co, -1, isz, isz + bc);
                let cc = slice_axis(co, -1, isz + bc, -1);
                let (y, _s) = m.ssm_step(&hs, &b, &cc, &dt, None);
                mlxcel_core::reshape(&y, &[1, 1, isz])
            }),
        ),
        (
            "m.gated_norm",
            Box::new(move |c: &Ctx| {
                let gate = slice_axis(&c["m.in_proj"], -1, 0, isz);
                m.norm.forward(&c["m.ssm_step"], &gate)
            }),
        ),
        (
            "m.out_proj",
            Box::new(move |c: &Ctx| m.out_proj.forward(&c["m.gated_norm"])),
        ),
    ]
}

fn attn_stages<'a>(layer: &'a GraniteMoeHybridDecoderLayer) -> Vec<Stage<'a>> {
    let Mixer::Attention(a) = &layer.mixer else {
        unreachable!()
    };
    vec![
        (
            "norm1",
            Box::new(move |c: &Ctx| layer.input_layernorm.forward(&c["x"])),
        ),
        (
            "a.attention",
            Box::new(move |c: &Ctx| {
                let mut kv = KVCache::new();
                a.forward(&c["norm1"], &mut kv, None)
            }),
        ),
    ]
}

fn ff_stages<'a>(
    layer: &'a GraniteMoeHybridDecoderLayer,
    mixer_key: &'static str,
) -> Vec<Stage<'a>> {
    let rm = layer.residual_multiplier;
    let mut v: Vec<Stage<'a>> = vec![
        (
            "resid1",
            Box::new(move |c: &Ctx| {
                let mo = mlxcel_core::multiply_scalar(&c[mixer_key], rm);
                mlxcel_core::add(&c["x"], &mo)
            }),
        ),
        (
            "norm2",
            Box::new(move |c: &Ctx| layer.post_attention_layernorm.forward(&c["resid1"])),
        ),
    ];
    let FeedForward::Moe { moe, shared } = &layer.feed_forward else {
        panic!("granite-4.0-h-tiny is MoE")
    };
    v.push((
        "f.router",
        Box::new(move |c: &Ctx| {
            let xf = mlxcel_core::reshape(&c["norm2"], &[1, -1]);
            moe.router.forward(&xf)
        }),
    ));
    v.push((
        "f.experts",
        Box::new(move |c: &Ctx| {
            let xf = mlxcel_core::reshape(&c["norm2"], &[1, -1]);
            moe.switch_mlp.forward(&xf, &c["idx"])
        }),
    ));
    v.push((
        "f.moe_sum",
        Box::new(move |c: &Ctx| {
            moe_weighted_sum(
                &c["f.experts"],
                &c["scores"],
                mlxcel_core::array_dtype(&c["norm2"]),
            )
        }),
    ));
    v.push((
        "f.shared",
        Box::new(move |c: &Ctx| shared.forward(&c["norm2"])),
    ));
    v.push((
        "out",
        Box::new(move |c: &Ctx| {
            let moe_o = mlxcel_core::reshape(&c["f.moe_sum"], &[1, 1, -1]);
            let ff = mlxcel_core::add(&moe_o, &c["f.shared"]);
            let ff = mlxcel_core::multiply_scalar(&ff, rm);
            mlxcel_core::add(&c["resid1"], &ff)
        }),
    ));
    v
}

fn route(
    moe: &GraniteMoeHybridMoE,
    logits: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>, Vec<i32>) {
    let kth = moe.num_experts - moe.top_k;
    let ind = mlxcel_core::argpartition(logits, kth, -1);
    let s = mlxcel_core::array_shape(&ind);
    let top = mlxcel_core::slice(&ind, &[0, kth], &[s[0], s[1]]);
    let tl = mlxcel_core::take_along_axis(logits, &top, -1);
    let tl = astype(&tl, dtype::FLOAT32);
    let sc = mlxcel_core::softmax(&tl, -1);
    eval(&top);
    eval(&sc);
    let mut ids: Vec<i32> = f32v(&top).iter().map(|x| *x as i32).collect();
    ids.sort();
    (top, sc, ids)
}

/// Full forward, recording the hidden state after every layer and the logits.
fn full_forward(
    model: &GraniteMoeHybridModel,
    tok: i32,
    f32_act: bool,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let input = from_slice_i32(&[tok], &[1, 1]);
    let mut h = model.embed_tokens.forward(&input);
    if f32_act {
        h = astype(&h, dtype::FLOAT32);
    }
    let mut h = mlxcel_core::multiply_scalar(&h, model.embedding_multiplier);
    let mut caches = model.make_caches();
    let mut hs = Vec::new();
    for (layer, cache) in model.layers.iter().zip(caches.iter_mut()) {
        h = layer.forward(&h, cache, None);
        hs.push(f32v(&h));
    }
    let hn = model.norm.forward(&h);
    let lg = model.embed_tokens.as_linear(&hn);
    let lg = mlxcel_core::divide_scalar(&lg, model.logits_scaling);
    (hs, f32v(&lg))
}

fn trace_row(c: usize, target: i32, lg: &[f32], topk: usize) -> String {
    let m = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + lg.iter().map(|x| ((*x as f64) - m).exp()).sum::<f64>().ln();
    let nll = lse - lg[target as usize] as f64;
    let mut idx: Vec<usize> = (0..lg.len()).collect();
    idx.sort_by(|a, b| lg[*b].total_cmp(&lg[*a]));
    let ids: Vec<String> = idx[..topk].iter().map(|i| i.to_string()).collect();
    let ls: Vec<String> = idx[..topk]
        .iter()
        .map(|i| format!("{:.6}", lg[*i]))
        .collect();
    format!(
        "{c}\t0\t{target}\t{nll:.6}\t{}\t{}",
        ids.join(","),
        ls.join(",")
    )
}

#[test]
#[ignore]
fn probe_2154() {
    let model_dir = std::env::var("PROBE_MODEL").unwrap();
    let corpus = std::env::var("PROBE_CORPUS").unwrap();
    let out = std::path::PathBuf::from(std::env::var("PROBE_OUT").unwrap());
    let n: usize = std::env::var("PROBE_N")
        .map(|s| s.parse().unwrap())
        .unwrap_or(128);
    let arms: String = std::env::var("PROBE_ARMS").unwrap_or_else(|_| "ops,f32cpu,f32gpu".into());
    std::fs::create_dir_all(&out).unwrap();

    let tok = crate::tokenizer::load_tokenizer(Path::new(&model_dir)).unwrap();
    let text = std::fs::read_to_string(&corpus).unwrap();
    let ids: Vec<i32> = tok
        .encode(&text, false)
        .unwrap()
        .iter()
        .map(|&t| t as i32)
        .collect();
    let (model, _) = GraniteMoeHybridModel::load(&model_dir).unwrap();

    // Trace arms in logit_trace's row format (topk 8, logits not bf16-rounded
    // for the f32 arms).
    for arm in ["f32cpu", "f32gpu", "bf16cpu", "bf16gpu"] {
        if !arms.split(',').any(|a| a == arm) {
            continue;
        }
        set_gpu(arm.ends_with("gpu"));
        let f32_act = arm.starts_with("f32");
        let mut f = std::fs::File::create(out.join(format!("probe_{arm}_w1.tsv"))).unwrap();
        writeln!(f, "# arm\t{arm}").unwrap();
        for c in 0..n {
            let (_, lg) = full_forward(&model, ids[c], f32_act);
            writeln!(f, "{}", trace_row(c, ids[c + 1], &lg, 8)).unwrap();
            mlxcel_core::clear_memory_cache();
        }
        eprintln!("arm {arm} done");
        f.flush().unwrap();
    }

    if arms.split(',').any(|a| a == "drift") {
        // Per-layer drift of the full forward: gpu bf16 and cpu bf16 against cpu f32.
        let mut drift_g: Vec<Acc> = vec![Acc::default(); model.layers.len()];
        let mut drift_c: Vec<Acc> = vec![Acc::default(); model.layers.len()];
        let mut drift_gc: Vec<Acc> = vec![Acc::default(); model.layers.len()];
        for c in 0..n {
            set_gpu(false);
            let (hf, _) = full_forward(&model, ids[c], true);
            let (hc, _) = full_forward(&model, ids[c], false);
            set_gpu(true);
            let (hg, _) = full_forward(&model, ids[c], false);
            for l in 0..model.layers.len() {
                drift_g[l].add(&hg[l], &hf[l]);
                drift_c[l].add(&hc[l], &hf[l]);
                drift_gc[l].add(&hg[l], &hc[l]);
            }
            mlxcel_core::clear_memory_cache();
        }
        let mut f = std::fs::File::create(out.join("probe_drift.tsv")).unwrap();
        writeln!(f, "layer\tkind\tgpu_vs_f32\tcpu_vs_f32\tgpu_vs_cpu").unwrap();
        for l in 0..model.layers.len() {
            let kind = if model.layers[l].is_attention() {
                "attn"
            } else {
                "mamba"
            };
            writeln!(
                f,
                "{l}\t{kind}\t{:.5}\t{:.5}\t{:.5}",
                drift_g[l].rel / n as f64,
                drift_c[l].rel / n as f64,
                drift_gc[l].rel / n as f64
            )
            .unwrap();
        }
    }

    if arms.split(',').any(|a| a == "ops") {
        // Local per-op error on the GPU: each stage in bf16 from the bf16
        // chain's inputs, against the same stage in f32 on the same inputs
        // cast to f32. Optional CPU f32 check of the reference on a few tokens.
        set_gpu(true);
        let mut acc: BTreeMap<String, Acc> = BTreeMap::new();
        let mut route_flips = 0usize;
        let mut route_total = 0usize;
        let ops_n: usize = std::env::var("PROBE_OPS_N")
            .map(|s| s.parse().unwrap())
            .unwrap_or(n);
        let t0 = std::time::Instant::now();
        for c in 0..ops_n {
            let input = from_slice_i32(&[ids[c]], &[1, 1]);
            let h0 = model.embed_tokens.forward(&input);
            let mut h = mlxcel_core::multiply_scalar(&h0, model.embedding_multiplier);
            eval(&h);
            for layer in model.layers.iter() {
                let mut stages = if layer.is_attention() {
                    attn_stages(layer)
                } else {
                    mamba_stages(layer)
                };
                let mk = if layer.is_attention() {
                    "a.attention"
                } else {
                    "m.out_proj"
                };
                stages.extend(ff_stages(layer, mk));
                let FeedForward::Moe { moe, .. } = &layer.feed_forward else {
                    unreachable!()
                };
                let mut ctx: Ctx = BTreeMap::new();
                put(&mut ctx, "x", mlxcel_core::copy(&h));
                for (name, f) in stages.iter() {
                    let o = f(&ctx);
                    put(&mut ctx, name, o);
                    if *name == "f.router" {
                        let (top, sc, ci) = route(moe, &ctx["f.router"]);
                        let (_, _, fi) = {
                            // f32 router logits from the same bf16 input.
                            let xf = mlxcel_core::reshape(
                                &astype(&ctx["norm2"], dtype::FLOAT32),
                                &[1, -1],
                            );
                            let l32 = moe.router.forward(&xf);
                            route(moe, &l32)
                        };
                        route_total += 1;
                        if fi != ci {
                            route_flips += 1;
                        }
                        put(&mut ctx, "idx", top);
                        put(&mut ctx, "scores", sc);
                    }
                }
                let mut ctx32: Ctx = BTreeMap::new();
                for (k, v) in ctx.iter() {
                    let a = if k == "idx" {
                        mlxcel_core::copy(v)
                    } else {
                        astype(v, dtype::FLOAT32)
                    };
                    ctx32.insert(k.clone(), a);
                }
                for (name, f) in stages.iter() {
                    let r = f32v(&f(&ctx32));
                    let g = f32v(&ctx[*name]);
                    acc.entry(name.to_string()).or_default().add(&g, &r);
                }
                h = mlxcel_core::copy(&ctx["out"]);
                eval(&h);
            }
            let lg = mlxcel_core::divide_scalar(
                &model.embed_tokens.as_linear(&model.norm.forward(&h)),
                model.logits_scaling,
            );
            let h32 = astype(&h, dtype::FLOAT32);
            let n32 = model.norm.forward(&h32);
            let lg32 = mlxcel_core::divide_scalar(
                &model.embed_tokens.as_linear(&n32),
                model.logits_scaling,
            );
            acc.entry("final_norm".into())
                .or_default()
                .add(&f32v(&model.norm.forward(&h)), &f32v(&n32));
            acc.entry("head".into())
                .or_default()
                .add(&f32v(&lg), &f32v(&lg32));
            let nb = astype(&n32, mlxcel_core::array_dtype(&h));
            let lgh = mlxcel_core::divide_scalar(
                &model.embed_tokens.as_linear(&nb),
                model.logits_scaling,
            );
            acc.entry("head_only".into())
                .or_default()
                .add(&f32v(&lgh), &f32v(&lg32));
            mlxcel_core::clear_memory_cache();
            if c % 32 == 0 {
                eprintln!("ops token {c} at {:.1}s", t0.elapsed().as_secs_f64());
            }
        }
        let mut f = std::fs::File::create(out.join("probe_ops.tsv")).unwrap();
        writeln!(
            f,
            "stage\tcount\tbf16_vs_f32_rel_mean\tbf16_vs_f32_rel_max\tabs_max"
        )
        .unwrap();
        for (k, a) in acc.iter() {
            writeln!(
                f,
                "{k}\t{}\t{:.6}\t{:.6}\t{:.6}",
                a.n,
                a.rel / a.n as f64,
                a.rel_max,
                a.abs_max
            )
            .unwrap();
        }
        writeln!(
            f,
            "# route_flips_vs_f32_router\t{route_flips}\t{route_total}"
        )
        .unwrap();
    }

    // Mixed-precision arms: f32 activations except the named families, which
    // run in bf16 (input cast to bf16, output cast back to f32).
    if let Ok(mix) = std::env::var("PROBE_MIX") {
        set_gpu(true);
        for spec in mix.split(';') {
            let fam: Vec<&str> = spec.split('+').collect();
            let mut f = std::fs::File::create(
                out.join(format!("probe_mix_{}_w1.tsv", spec.replace('+', "-"))),
            )
            .unwrap();
            writeln!(f, "# mix\t{spec}").unwrap();
            for c in 0..n {
                let lg = mixed_forward(&model, ids[c], &fam);
                writeln!(f, "{}", trace_row(c, ids[c + 1], &lg, 8)).unwrap();
                mlxcel_core::clear_memory_cache();
            }
            eprintln!("mix {spec} done");
        }
    }
}

fn mixed_forward(model: &GraniteMoeHybridModel, tok: i32, fam: &[&str]) -> Vec<f32> {
    let on = |k: &str| fam.contains(&k) || fam.contains(&"all");
    let bf = |a: &MlxArray| astype(a, dtype::BFLOAT16);
    let f3 = |a: &MlxArray| astype(a, dtype::FLOAT32);
    let wrap = |k: &str, x: &MlxArray, g: &dyn Fn(&MlxArray) -> UniquePtr<MlxArray>| {
        if on(k) { f3(&g(&bf(x))) } else { g(x) }
    };
    let input = from_slice_i32(&[tok], &[1, 1]);
    let h = f3(&model.embed_tokens.forward(&input));
    let mut h = mlxcel_core::multiply_scalar(&h, model.embedding_multiplier);
    if on("resid") {
        h = f3(&bf(&h));
    }
    for layer in model.layers.iter() {
        let normed = wrap("norm", &h, &|x| layer.input_layernorm.forward(x));
        let mo = match &layer.mixer {
            Mixer::Mamba(m) => wrap("mamba", &normed, &|x| m.forward(x, None)),
            Mixer::Attention(a) => wrap("attn", &normed, &|x| {
                let mut kv = KVCache::new();
                a.forward(x, &mut kv, None)
            }),
        };
        let mo = mlxcel_core::multiply_scalar(&mo, layer.residual_multiplier);
        h = mlxcel_core::add(&h, &mo);
        if on("resid") {
            h = f3(&bf(&h));
        }
        let n2 = wrap("norm", &h, &|x| layer.post_attention_layernorm.forward(x));
        let ff = match &layer.feed_forward {
            FeedForward::Moe { moe, shared } => {
                let a = wrap("moe", &n2, &|x| moe.forward(x));
                let b = wrap("shared", &n2, &|x| shared.forward(x));
                mlxcel_core::add(&a, &b)
            }
            other => other.forward(&n2),
        };
        let ff = mlxcel_core::multiply_scalar(&ff, layer.residual_multiplier);
        h = mlxcel_core::add(&h, &ff);
        if on("resid") {
            h = f3(&bf(&h));
        }
    }
    let hn = wrap("norm", &h, &|x| model.norm.forward(x));
    let lg = wrap("head", &hn, &|x| {
        mlxcel_core::divide_scalar(&model.embed_tokens.as_linear(x), model.logits_scaling)
    });
    f32v(&lg)
}
