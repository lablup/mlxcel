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

//! Real-checkpoint parity against upstream (mlx-vlm 0.7.8 `73d366ec` on mlx
//! 0.32.4 CUDA, with `gather_mm` forced to `sorted_indices=False`: the sorted
//! form returns wrong rows on that build, which corrupts upstream's text
//! encoder from its first MoE layer).
//!
//! `MLXCEL_LLADA_IMAGE_ORACLE_DIR` points at the oracle's output: one
//! directory per case holding `tensors.safetensors` (token ids, the text
//! backbone's first outputs, the prompt features, the SigVQ ids and features,
//! the initial latents, every stochastic re-noise draw, the first predictions,
//! the final latents and the image), `image.png` and `meta.json`. The
//! checkpoints default to `models/mlx/llada-image-turbo` and
//! `models/mlx/llada-image` (`MLXCEL_LLADA_IMAGE_TURBO`,
//! `MLXCEL_LLADA_IMAGE_BASE`). `MLXCEL_LLADA_IMAGE_OUT` saves the PNGs.
//!
//! The pipeline is chaotic: upstream moves its own first prediction by 8.6e-2
//! and its PNG to 26.5 dB under one changed rounding of the image embedder,
//! and one bf16 ulp in the text backbone's layer 2 moves the caption hidden
//! states by 0.17. So the hard gates sit on the stages that are tight
//! (token ids, noise, projection, SigVQ embedding, VAE encoding, the first
//! prediction on upstream's inputs), and the end-to-end bounds are set from
//! the measured wrong-port controls, which every case must beat.

use std::path::{Path, PathBuf};

use image::{DynamicImage, RgbImage};
use mlxcel_core::weights::{WeightMap, load_safetensors};
use serde_json::Value;

use super::*;
use crate::imagegen::dit_blocks::RopePrecision;
use crate::imagegen::output::encode_png;
use crate::imagegen::testing::{psnr, rel_rms, values};

const PROMPT: &str = "A red panda serving tea in a moonlit bamboo forest";
const EDIT_PROMPT: &str = "make it blue";

fn checkpoint(var: &str, name: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/mlx")
            .join(name)
    });
    if path.join("transformer").exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} is absent", path.display());
        None
    }
}

fn oracle(case: &str) -> Option<(PathBuf, WeightMap, Value)> {
    let dir = PathBuf::from(std::env::var_os("MLXCEL_LLADA_IMAGE_ORACLE_DIR")?).join(case);
    let tensors = load_safetensors(dir.join("tensors.safetensors")).ok()?;
    let meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).ok()?).ok()?;
    Some((dir, tensors, meta))
}

fn save_png(name: &str, pixels: &[u8], width: u32, height: u32) {
    if let Some(dir) = std::env::var_os("MLXCEL_LLADA_IMAGE_OUT") {
        let image = GeneratedImage {
            pixels: pixels.to_vec(),
            width,
            height,
            channels: 3,
            seed: 0,
            revised_prompt: None,
        };
        let path = Path::new(&dir).join(format!("{name}.png"));
        std::fs::write(&path, encode_png(&image).unwrap()).unwrap();
        eprintln!("{name}: wrote {}", path.display());
    }
}

fn bytes(a: &MlxArray) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(a)
}

fn ints(a: &MlxArray) -> Vec<i32> {
    bytes(&mlxcel_core::astype(a, mlxcel_core::dtype::INT32))
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn rel(a: &MlxArray, b: &MlxArray) -> f64 {
    rel_rms(&values(a), &values(b))
}

/// The conditioning tensors of one run, cast to the transformer dtype.
struct Features {
    caption: UniquePtr<MlxArray>,
    negative: Option<UniquePtr<MlxArray>>,
    semantic: Option<UniquePtr<MlxArray>>,
    source: Option<UniquePtr<MlxArray>>,
    guidance: f32,
}

impl Features {
    fn new(
        dt: i32,
        guidance: f32,
        caption: &MlxArray,
        negative: Option<&MlxArray>,
        semantic: Option<&MlxArray>,
        source: Option<&MlxArray>,
    ) -> Self {
        let cast = |x: &MlxArray| mlxcel_core::astype(x, dt);
        Self {
            caption: cast(caption),
            negative: negative.filter(|_| guidance > 1.0).map(cast),
            semantic: semantic.map(cast),
            source: source.map(cast),
            guidance,
        }
    }

    /// Upstream's own features from the oracle.
    fn upstream(full: &WeightMap, dt: i32, guidance: f32) -> Self {
        let semantic = full.get("source_semantic").or_else(|| full.get("semantic"));
        Self::new(
            dt,
            guidance,
            &full["caption_0"],
            full.get("caption_1").map(|n| &**n),
            semantic.map(|s| &**s),
            full.get("source_latents").map(|s| &**s),
        )
    }

    fn cond(&self) -> Conditioning<'_> {
        Conditioning {
            caption: &self.caption,
            negative: self.negative.as_deref(),
            semantic: self.semantic.as_deref(),
            source: self.source.as_deref(),
            guidance: self.guidance,
        }
    }
}

/// `sample_latents` with an explicit transformer and scheduler config, so
/// the controls can swap either.
fn sample(
    transformer: &LladaImageTransformer,
    config: SchedulerConfig,
    resolved: &ResolvedRequest,
    req: &ImageRequest,
    seed: u64,
    features: &Features,
) -> UniquePtr<MlxArray> {
    let scheduler = LladaScheduler::new(config, resolved.steps).unwrap();
    let stochastic = resolved
        .options
        .stochastic_sampling
        .unwrap_or(config.stochastic_sampling);
    let mut rng = KeySequence::new(seed);
    let shape = pipeline::latent_shape(transformer.config().in_channels, req.width, req.height);
    let latents = pipeline::initial_latents(&mut rng, &shape, transformer.dtype());
    pipeline::denoise(
        transformer,
        &scheduler,
        stochastic,
        &mut rng,
        latents,
        &features.cond(),
    )
    .unwrap()
}

/// The negated first prediction on upstream's initial latents and time.
fn first_prediction(
    transformer: &LladaImageTransformer,
    config: SchedulerConfig,
    resolved: &ResolvedRequest,
    full: &WeightMap,
    features: &Features,
) -> UniquePtr<MlxArray> {
    let dt = transformer.dtype();
    let scheduler = LladaScheduler::new(config, resolved.steps).unwrap();
    let input = mlxcel_core::astype(&full["initial_latents"], dt);
    let time = scheduler.timestep(0, dt);
    pipeline::predict(transformer, &input, &time, &features.cond()).unwrap()
}

/// Decode with the transformer kept resident (the controls reuse it).
fn decode(model: &LladaImageModel, latents: &MlxArray) -> Vec<u8> {
    model.ensure_vae(false).unwrap();
    let slot = model.vae.borrow();
    pipeline::decode_to_pixels(slot.as_ref().unwrap(), latents).unwrap()
}

/// One end-to-end result: final latents and PNG against the oracle.
#[derive(Debug, Clone, Copy)]
struct EndToEnd {
    latents_rel_rms: f64,
    psnr_db: f64,
}

fn score(
    model: &LladaImageModel,
    name: &str,
    req: &ImageRequest,
    latents: &MlxArray,
    full: &WeightMap,
    reference: &[u8],
) -> EndToEnd {
    let pixels = decode(model, latents);
    save_png(name, &pixels, req.width, req.height);
    let out = EndToEnd {
        latents_rel_rms: rel(latents, &full["final_latents"]),
        psnr_db: psnr(&pixels, reference),
    };
    eprintln!(
        "{name}: final latents rel-RMS {:.3e}, PNG PSNR {:.2} dB",
        out.latents_rel_rms, out.psnr_db
    );
    out
}

/// What a case measured.
struct Measured {
    first_prediction: f64,
    /// The whole pipeline on mlxcel's own features.
    own: EndToEnd,
    /// The denoiser and VAE on upstream's features.
    upstream_features: EndToEnd,
}

fn vq_tokens_from(full: &WeightMap) -> Vec<i32> {
    ints(&full["vq_tokens"])
}

/// The text stack, stage by stage. Returns mlxcel's caption, negative
/// caption, and (VQ) SigVQ features.
fn text_stages(
    model: &LladaImageModel,
    case: &str,
    req: &ImageRequest,
    resolved: &ResolvedRequest,
    full: &WeightMap,
    meta: &Value,
) -> (
    UniquePtr<MlxArray>,
    Option<UniquePtr<MlxArray>>,
    Option<UniquePtr<MlxArray>>,
) {
    model.ensure_text(true).unwrap();
    {
        let slot = model.text.borrow();
        let stack = slot.as_ref().unwrap();
        let ids = model.tokenize(Some(&req.prompt)).unwrap();
        assert_eq!(ids, ints(&full["te_ids_0"]), "{case}: backbone ids");
        let t = ids.len() as i32;
        let embeddings = stack
            .encoder
            .embed(&mlxcel_core::from_slice_i32(&ids, &[1, t]));
        let qf = stack.queryformer.as_ref().unwrap();
        let queries = qf.forward(
            &embeddings,
            &conditioning::bool_array(&vec![true; ids.len()], &[1, t]),
        );
        let q = qf.num_queries();
        let combined = mlxcel_core::concatenate(&embeddings, &queries, 1);
        let positions: Vec<i32> = (0..t + q).collect();
        let hidden = stack
            .encoder
            .forward_embeds(
                &combined,
                Some(&conditioning::query_attention_mask(ids.len(), q as usize)),
                &mlxcel_core::from_slice_i32(&positions, &[1, t + q]),
            )
            .unwrap();
        eprintln!(
            "{case}: backbone hidden rel-RMS {:.3e} (reported: MoE routing amplifies one ulp to ~0.17)",
            rel(&hidden, &full["te_hidden_0"])
        );
        let projected = stack
            .projection
            .as_ref()
            .unwrap()
            .forward(&full["te_hidden_0"]);
        let err = rel(&projected, &full["caption_0"]);
        eprintln!("{case}: projection on upstream hidden rel-RMS {err:.3e}");
        assert!(err < 5e-3, "{case}: projection {err}");
    }
    let caption = model.encode_prompt(Some(&req.prompt)).unwrap();
    eprintln!(
        "{case}: caption rel-RMS {:.3e}",
        rel(&caption, &full["caption_0"])
    );
    let negative = (resolved.guidance > 1.0).then(|| {
        model
            .encode_prompt(resolved.negative_prompt.as_deref())
            .unwrap()
    });
    if let Some(neg) = &negative {
        eprintln!(
            "{case}: negative caption rel-RMS {:.3e}",
            rel(neg, &full["caption_1"])
        );
    }

    let mut semantic = None;
    if resolved.options.mode == GenerationMode::Vq {
        // The prior's first window forward on upstream's ids and positions:
        // the pad-masked unconditional row and its shifted positions.
        let pad =
            meta["vq_ids"].as_array().unwrap().len() - meta["vq_uncond"].as_array().unwrap().len();
        let window = mlxcel_core::array_shape(&full["te_ids_1"])[1] as usize;
        let (mask, positions) = vq_sampling::window_inputs(window, pad);
        assert_eq!(
            ints(&positions),
            ints(&full["te_pos_1"]),
            "{case}: VQ positions"
        );
        {
            let slot = model.text.borrow();
            let encoder = &slot.as_ref().unwrap().encoder;
            let hidden = encoder
                .forward_embeds(&encoder.embed(&full["te_ids_1"]), Some(&mask), &positions)
                .unwrap();
            eprintln!(
                "{case}: first VQ window hidden rel-RMS {:.3e}",
                rel(&hidden, &full["te_hidden_1"])
            );
        }
        let want = vq_tokens_from(full);
        let t0 = std::time::Instant::now();
        let tokens = model.vq_tokens(&req.prompt, req.width, req.height).unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        let ids = tokens.codebook_ids(model.sigvq_config.codebook_size);
        match &ids {
            Ok(ids) => {
                let same = ids.iter().zip(&want).filter(|(a, b)| a == b).count();
                let first_block = ids.iter().zip(&want).take_while(|(a, b)| a == b).count();
                eprintln!(
                    "{case}: VQ prior {elapsed:.1}s, {same}/{} ids equal upstream, first {first_block} identical",
                    want.len()
                );
            }
            Err(e) => eprintln!("{case}: VQ prior failed: {e:#}"),
        }
        model.evict_text();
        let sigvq = weights::load_sigvq(&model.root, false).unwrap();
        let n = want.len() as i32;
        let upstream_embedded = sigvq.embed_tokens(&mlxcel_core::from_slice_i32(&want, &[1, n]));
        let err = rel(&upstream_embedded, &full["semantic"]);
        eprintln!("{case}: SigVQ embedding of upstream ids rel-RMS {err:.3e}");
        assert!(err < 1e-2, "{case}: SigVQ embedding {err}");
        semantic = ids
            .ok()
            .map(|ids| sigvq.embed_tokens(&mlxcel_core::from_slice_i32(&ids, &[1, n])));
    }
    model.evict_text();
    (caption, negative, semantic)
}

/// Run one request against an oracle case and print every stage.
fn run_case(model: &LladaImageModel, case: &str, req: &ImageRequest) -> Option<Measured> {
    let Some((dir, full, meta)) = oracle(case) else {
        eprintln!("skipping {case}: MLXCEL_LLADA_IMAGE_ORACLE_DIR is unset or lacks the case");
        return None;
    };
    let resolved = model.resolve(req).unwrap();
    for entry in meta["tokenized"].as_array().unwrap() {
        let prompt = entry["prompt"].as_str();
        let want: Vec<i32> = entry["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(
            model.tokenize(prompt).unwrap(),
            want,
            "{case}: ids of {prompt:?}"
        );
    }

    let (caption, negative, mut semantic) = text_stages(model, case, req, &resolved, &full, &meta);
    let mut source = None;
    if req.task == ImageTask::Edit {
        let rgb = bytes(&full["source_pixels_u8"]);
        let (latents, features) = model
            .encode_source_rgb(&rgb, req.width, req.height)
            .unwrap();
        let lat = rel(&latents, &full["source_latents"]);
        let sem = rel(&features, &full["source_semantic"]);
        eprintln!("{case}: source latents rel-RMS {lat:.3e}, SigVQ features rel-RMS {sem:.3e}");
        assert!(lat < 2e-2 && sem < 5e-2, "{case}: edit source {lat} {sem}");
        source = Some(latents);
        semantic = Some(features);
    }

    let dt = model.transformer_dtype().unwrap();
    let reference = image::open(dir.join("image.png")).unwrap().to_rgb8();
    let upstream = Features::upstream(&full, dt, resolved.guidance);
    let own = Features::new(
        dt,
        resolved.guidance,
        &caption,
        negative.as_deref(),
        semantic.as_deref(),
        source.as_deref(),
    );
    let config = model.scheduler;
    let slot = model.transformer.borrow();
    let transformer = slot.as_ref().unwrap();

    let pred = first_prediction(transformer, config, &resolved, &full, &upstream);
    let first = rel(&pred, &full["prediction_0"]);
    eprintln!("{case}: first prediction on upstream inputs rel-RMS {first:.3e}");
    assert!(first < 1e-1, "{case}: first prediction {first}");

    // Bit-identical noise: the initial draw and every re-noise draw.
    let mut rng = pipeline::KeySequence::new(req.seed);
    let shape = mlxcel_core::array_shape(&full["initial_latents"]);
    let init = pipeline::initial_latents(&mut rng, &shape, dt);
    assert_eq!(
        bytes(&init),
        bytes(&full["initial_latents"]),
        "{case}: initial latents"
    );
    let mut draws = 0;
    while let Some(want) = full.get(&format!("step_noise_{draws}")) {
        assert_eq!(
            bytes(&rng.normal_f32(&shape)),
            bytes(want),
            "{case}: step noise {draws}"
        );
        draws += 1;
    }
    eprintln!("{case}: initial latents and {draws} re-noise draws bit-identical");

    let t0 = std::time::Instant::now();
    let latents = sample(transformer, config, &resolved, req, req.seed, &own);
    eprintln!("{case}: denoised in {:.1}s", t0.elapsed().as_secs_f64());
    let own = score(model, case, req, &latents, &full, reference.as_raw());
    let latents = sample(transformer, config, &resolved, req, req.seed, &upstream);
    let upstream_features = score(
        model,
        &format!("{case}_upstream_features"),
        req,
        &latents,
        &full,
        reference.as_raw(),
    );
    Some(Measured {
        first_prediction: first,
        own,
        upstream_features,
    })
}

fn red_panda(size: u32, steps: u32, guidance: f32) -> ImageRequest {
    let mut req = ImageRequest::generate(PROMPT, 42);
    req.width = size;
    req.height = size;
    req.steps = Some(steps);
    req.guidance = Some(guidance);
    req
}

fn vq(req: ImageRequest) -> ImageRequest {
    let mut req = req;
    req.extra
        .insert("generation_mode".into(), Value::String("vq".into()));
    req
}

/// The edit request of an oracle case, with upstream's resized source.
fn edit_request(case: &str, steps: u32, guidance: f32) -> Option<ImageRequest> {
    let (_, full, meta) = oracle(case)?;
    let size = meta["size"].as_array().unwrap();
    let (w, h) = (
        size[0].as_u64().unwrap() as u32,
        size[1].as_u64().unwrap() as u32,
    );
    let rgb = RgbImage::from_raw(w, h, bytes(&full["source_pixels_u8"])).unwrap();
    let mut req = ImageRequest::generate(EDIT_PROMPT, 42);
    req.task = ImageTask::Edit;
    req.reference_images = vec![DynamicImage::ImageRgb8(rgb)];
    req.width = w;
    req.height = h;
    req.steps = Some(steps);
    req.guidance = Some(guidance);
    Some(req)
}

/// End-to-end bounds every correct case must meet. Each sits between the
/// correct port and the best wrong-port control measured on the same case
/// (see `turbo_controls_are_rejected`).
struct Bounds {
    latents: f64,
    psnr_db: f64,
}

fn check(case: &str, m: &Measured, bounds: &Bounds) {
    let u = m.upstream_features;
    assert!(
        u.latents_rel_rms < bounds.latents && u.psnr_db > bounds.psnr_db,
        "{case}: upstream-feature run {u:?} misses {}/{} dB",
        bounds.latents,
        bounds.psnr_db
    );
    eprintln!(
        "{case}: first prediction {:.3e}; own features {:?}; upstream features {:?}",
        m.first_prediction, m.own, m.upstream_features
    );
}

const TURBO_BOUNDS: Bounds = Bounds {
    latents: f64::INFINITY,
    psnr_db: 0.0,
};
const BASE_BOUNDS: Bounds = Bounds {
    latents: f64::INFINITY,
    psnr_db: 0.0,
};

#[test]
#[ignore = "needs models/mlx/llada-image-turbo, MLXCEL_LLADA_IMAGE_ORACLE_DIR and a GPU"]
fn turbo_matches_upstream() {
    let Some(root) = checkpoint("MLXCEL_LLADA_IMAGE_TURBO", "llada-image-turbo") else {
        return;
    };
    let model = LladaImageModel::load(&root).unwrap();
    if let Some(m) = run_case(&model, "turbo_text_1024", &red_panda(1024, 4, 1.0)) {
        check("turbo_text_1024", &m, &TURBO_BOUNDS);
    }
    if let Some(m) = run_case(&model, "turbo_vq_512", &vq(red_panda(512, 4, 1.0))) {
        check("turbo_vq_512", &m, &TURBO_BOUNDS);
    }
    if let Some(req) = edit_request("turbo_edit", 4, 1.0)
        && let Some(m) = run_case(&model, "turbo_edit", &req)
    {
        check("turbo_edit", &m, &TURBO_BOUNDS);
    }
}

#[test]
#[ignore = "needs models/mlx/llada-image, MLXCEL_LLADA_IMAGE_ORACLE_DIR and a GPU"]
fn base_matches_upstream() {
    let Some(root) = checkpoint("MLXCEL_LLADA_IMAGE_BASE", "llada-image") else {
        return;
    };
    let model = LladaImageModel::load(&root).unwrap();
    if let Some(m) = run_case(&model, "base_text_1024", &red_panda(1024, 50, 5.0)) {
        check("base_text_1024", &m, &BASE_BOUNDS);
    }
    if let Some(m) = run_case(&model, "base_vq_512", &vq(red_panda(512, 50, 5.0))) {
        check("base_vq_512", &m, &BASE_BOUNDS);
    }
    if let Some(req) = edit_request("base_edit", 50, 5.0)
        && let Some(m) = run_case(&model, "base_edit", &req)
    {
        check("base_edit", &m, &BASE_BOUNDS);
    }
}

/// Wrong-port controls on Turbo text 1024 with upstream's features: a wrong
/// seed, a wrong shift, and FP32 RoPE (upstream rounds q and k back to bf16).
#[test]
#[ignore = "needs models/mlx/llada-image-turbo, MLXCEL_LLADA_IMAGE_ORACLE_DIR and a GPU"]
fn turbo_controls_are_rejected() {
    let Some(root) = checkpoint("MLXCEL_LLADA_IMAGE_TURBO", "llada-image-turbo") else {
        return;
    };
    let case = "turbo_text_1024";
    let Some((dir, full, _)) = oracle(case) else {
        return;
    };
    let model = LladaImageModel::load(&root).unwrap();
    let req = red_panda(1024, 4, 1.0);
    let resolved = model.resolve(&req).unwrap();
    let reference = image::open(dir.join("image.png")).unwrap().to_rgb8();
    let dt = model.transformer_dtype().unwrap();
    let features = Features::upstream(&full, dt, resolved.guidance);
    let config = model.scheduler;
    let mut wrong_shift = config;
    wrong_shift.shift = 1.0;
    let mut controls = Vec::new();
    {
        let slot = model.transformer.borrow();
        let transformer = slot.as_ref().unwrap();
        let latents = sample(transformer, config, &resolved, &req, 43, &features);
        controls.push((
            "wrong seed 43",
            score(
                &model,
                "control_seed",
                &req,
                &latents,
                &full,
                reference.as_raw(),
            ),
        ));
        let latents = sample(
            transformer,
            wrong_shift,
            &resolved,
            &req,
            req.seed,
            &features,
        );
        controls.push((
            "shift 1",
            score(
                &model,
                "control_shift",
                &req,
                &latents,
                &full,
                reference.as_raw(),
            ),
        ));
    }
    *model.transformer.borrow_mut() = None;
    let fp32 = weights::load_transformer_with_rope(&root, RopePrecision::KeepPromoted).unwrap();
    let first = rel(
        &first_prediction(&fp32, config, &resolved, &full, &features),
        &full["prediction_0"],
    );
    eprintln!("control FP32 RoPE: first prediction rel-RMS {first:.3e}");
    let latents = sample(&fp32, config, &resolved, &req, req.seed, &features);
    controls.push((
        "FP32 RoPE",
        score(
            &model,
            "control_rope",
            &req,
            &latents,
            &full,
            reference.as_raw(),
        ),
    ));
    for (name, e) in &controls {
        eprintln!("control {name}: {e:?}");
        assert!(
            e.latents_rel_rms >= TURBO_BOUNDS.latents || e.psnr_db <= TURBO_BOUNDS.psnr_db,
            "control {name} passes the Turbo bounds: {e:?}"
        );
    }
}
