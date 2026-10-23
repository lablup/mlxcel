// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Deterministic two-stage flow sampling and object-local geometry generation.
use super::super::{DenseOutput, predictor::DenseSidecars};
use super::{
    bundle::{self, Model},
    config::Config,
    flow::State,
    processing,
    sparse::Grid,
    vision::FeatureCache,
};
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, MlxArray, UniquePtr};
use serde_json::{Map, Value, json};
use std::{collections::BTreeSet, path::Path};
type Array = UniquePtr<MlxArray>;
const LATENTS: [(&str, i32); 5] = [
    ("6drotation_normalized", 6),
    ("scale", 3),
    ("shape", 8),
    ("translation", 3),
    ("translation_scale", 1),
];
#[derive(Debug)]
pub struct Options {
    pub seed: u64,
    pub ss_steps: usize,
    pub slat_steps: usize,
    pub formats: Vec<String>,
    pub estimate_depth: bool,
}
impl Options {
    pub fn parse(c: &Config, o: &Map<String, Value>) -> Result<Self> {
        for key in o.keys() {
            ensure!(
                [
                    "task",
                    "seed",
                    "ss_steps",
                    "slat_steps",
                    "formats",
                    "estimate_depth"
                ]
                .contains(&key.as_str()),
                "unsupported Objects option {key}"
            );
        }
        let count = |k: &str, d: usize| -> Result<usize> {
            let n = match o.get(k) {
                None => d,
                Some(v) => usize::try_from(
                    v.as_u64()
                        .with_context(|| format!("{k} must be a positive integer"))?,
                )?,
            };
            ensure!(n > 0 && n <= 10000, "{k} must be in 1..=10000");
            Ok(n)
        };
        let seed = o
            .get("seed")
            .map(|v| v.as_u64().context("seed must be an unsigned integer"))
            .transpose()?
            .unwrap_or(42);
        let formats = match o.get("formats") {
            None => vec!["gaussian".into(), "mesh".into()],
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .context("formats entries must be strings")
                })
                .collect::<Result<Vec<_>>>()?,
            Some(_) => anyhow::bail!("formats must be an array"),
        };
        ensure!(
            !formats.is_empty()
                && formats
                    .iter()
                    .all(|s| ["gaussian", "gaussian_4", "mesh"].contains(&s.as_str())),
            "formats must contain gaussian, gaussian_4, or mesh"
        );
        ensure!(
            formats.iter().collect::<BTreeSet<_>>().len() == formats.len(),
            "formats must not contain duplicates"
        );
        let estimate_depth = o
            .get("estimate_depth")
            .map(|v| v.as_bool().context("estimate_depth must be boolean"))
            .transpose()?
            .unwrap_or(true);
        Ok(Self {
            seed,
            ss_steps: count("ss_steps", c.ss_steps)?,
            slat_steps: count("slat_steps", c.slat_steps)?,
            formats,
            estimate_depth,
        })
    }
}
fn schedule(n: usize, r: f64) -> Vec<f64> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            t / (1.0 + (r - 1.0) * (1.0 - t))
        })
        .collect()
}
fn normal(keys: &MlxArray, index: i32, shape: &[i32]) -> Array {
    let key = mx::reshape(&mx::slice(keys, &[index, 0], &[index + 1, 2]), &[2]);
    // SAFETY: the explicit MLX key remains alive throughout the synchronous graph-construction call.
    unsafe { mx::random_normal(shape, mx::dtype::FLOAT32, &*key) }
}
fn update(x: &MlxArray, v: &MlxArray, u: Option<&MlxArray>, strength: f32, dt: f64) -> Array {
    let v = mx::astype(v, mx::dtype::FLOAT32);
    let v = if let Some(u) = u {
        mx::subtract(
            &mx::multiply(&mx::from_slice_f32(&[1.0 + strength], &[]), &v),
            &mx::multiply(
                &mx::from_slice_f32(&[strength], &[]),
                &mx::astype(u, mx::dtype::FLOAT32),
            ),
        )
    } else {
        v
    };
    mx::add(x, &mx::multiply(&mx::from_slice_f32(&[dt as f32], &[]), &v))
}

pub fn occupancy_grid(logits: &MlxArray, prune: usize) -> Result<(Grid, usize)> {
    let s = mx::array_shape(logits);
    ensure!(
        s.len() == 5 && s[0] == 1 && s[1] == s[2] && s[2] == s[3] && s[4] == 1,
        "occupancy logits must be [1,R,R,R,1]"
    );
    let r = s[1];
    let values = mx::utils::array_to_vec_f32(logits);
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "non-finite occupancy logits"
    );
    let mut coords: Vec<[i32; 4]> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            (*v > 0.0).then_some([0, i as i32 / (r * r), i as i32 / r % r, i as i32 % r])
        })
        .collect();
    ensure!(
        !coords.is_empty(),
        "structure decoder produced no occupied voxels"
    );
    if prune > 0 {
        ensure!(prune == 1, "only one-voxel occupancy pruning is supported");
        let set: BTreeSet<_> = coords.iter().copied().collect();
        coords.retain(|c| {
            (-1..=1).any(|x| {
                (-1..=1)
                    .any(|y| (-1..=1).any(|z| !set.contains(&[0, c[1] + x, c[2] + y, c[3] + z])))
            })
        });
    }
    let mut factor = 1;
    if coords.len() > 42000 {
        let mut low = [i32::MAX; 3];
        let mut high = [i32::MIN; 3];
        for p in &coords {
            for j in 0..3 {
                low[j] = low[j].min(p[j + 1]);
                high[j] = high[j].max(p[j + 1]);
            }
        }
        for p in &mut coords {
            for j in 0..3 {
                let size = (high[j] - low[j] + 1) as f32;
                let target = size / 2.0;
                let start = low[j] as f32 + (size - target) / 2.0;
                let end = start + target - 1.0;
                p[j + 1] = (((p[j + 1] - low[j]) as f32 / (high[j] - low[j]).max(1) as f32
                    * (target - 1.0)
                    + start)
                    .round_ties_even()
                    .clamp(start as i32 as f32, end as i32 as f32))
                    as i32;
            }
        }
        coords.sort_unstable();
        coords.dedup();
        factor = 2;
        ensure!(
            coords.len() <= 42000,
            "occupancy remains too large after upstream downsampling"
        );
    }
    let flat: Vec<i32> = coords.iter().flatten().copied().collect();
    Ok((
        Grid::new(mx::from_slice_i32(&flat, &[coords.len() as i32, 4]), r)?,
        factor,
    ))
}

pub fn run(
    path: &Path,
    c: &Config,
    image: &image::DynamicImage,
    sidecars: &DenseSidecars,
    opts: &Map<String, Value>,
) -> Result<DenseOutput> {
    let options = Options::parse(c, opts)?;
    processing::validate_image_mask(image, sidecars.mask.as_ref())?;
    super::checkpoint_schema::validate_headers(path, c)?;
    let estimated = if sidecars.pointmap.is_none() && options.estimate_depth {
        bundle::estimate_depth(path, c, image)?
    } else {
        None
    };
    let pointmap = sidecars
        .pointmap
        .as_ref()
        .or(estimated.as_ref())
        .map(|v| &**v);
    let (inputs, metadata) =
        processing::prepare_inputs(image, sidecars.mask.as_ref(), pointmap, c.image_size)?;
    drop(estimated);
    eprintln!("sam3d_objects: loading main model");
    let model = Model::load(path, c)?;
    let keys = mx::random_split_key(&mx::random_key(options.seed), 6);
    let mut cache = FeatureCache::new();
    let mut state: State = LATENTS
        .iter()
        .enumerate()
        .map(|(i, (name, ch))| {
            (
                (*name).to_string(),
                normal(
                    &keys,
                    i as i32,
                    &[
                        1,
                        if *name == "shape" {
                            c.latent_resolution.pow(3) as i32
                        } else {
                            1
                        },
                        *ch,
                    ],
                ),
            )
        })
        .collect();
    {
        let tokens = model
            .structure_condition
            .forward(&model.backbones, &inputs, &mut cache)?;
        let condition = model.structure.prepare_condition(Some(&tokens))?;
        let uncondition = if c.ss_guidance > 0.0 {
            Some(model.structure.prepare_condition(None)?)
        } else {
            None
        };
        let times = schedule(options.ss_steps, c.ss_rescale_t);
        for (i, t) in times.windows(2).enumerate() {
            let time = mx::from_slice_f32(&[(t[0] * 1000.0) as f32], &[1]);
            let delta = mx::zeros(&[1], mx::dtype::FLOAT32);
            let guided = c.ss_guidance > 0.0 && t[0] <= 0.5;
            let (velocity, uncond) = if guided {
                let (a, b) = model.structure.guided(
                    &state,
                    &time,
                    &condition,
                    uncondition
                        .as_ref()
                        .context("missing unconditional structure context")?,
                    Some(&delta),
                )?;
                (a, Some(b))
            } else {
                (
                    model
                        .structure
                        .forward(&state, &time, &condition, Some(&delta))?,
                    None,
                )
            };
            state = state
                .iter()
                .map(|(n, x)| {
                    Ok((
                        n.clone(),
                        update(
                            x,
                            velocity.get(n).context("structure velocity missing")?,
                            uncond.as_ref().and_then(|u| u.get(n)).map(|v| &**v),
                            c.ss_guidance,
                            t[1] - t[0],
                        ),
                    ))
                })
                .collect::<Result<State>>()?;
            for value in state.values() {
                mx::eval(value);
            }
            eprintln!("sam3d_objects: structure {}/{}", i + 1, options.ss_steps);
        }
    }
    let (grid, factor) = occupancy_grid(
        &*model
            .occupancy
            .forward(state.get("shape").context("shape state missing")?)?,
        c.downsample_ss_dist,
    )?;
    let pose = super::pose::decode_pose(&state, &metadata, factor)?;
    drop(state);
    eprintln!(
        "sam3d_objects: occupancy {}",
        mx::array_shape(&grid.coords)[0]
    );
    let mut latent = normal(
        &keys,
        5,
        &[mx::array_shape(&grid.coords)[0], c.latent_channels as i32],
    );
    {
        let tokens = model
            .latent_condition
            .forward(&model.backbones, &inputs, &mut cache)?;
        let condition = model.latent.prepare_condition(Some(&tokens))?;
        let uncondition = if c.slat_guidance > 0.0 {
            Some(model.latent.prepare_condition(None)?)
        } else {
            None
        };
        drop(cache);
        drop(inputs);
        let times = schedule(options.slat_steps, c.slat_rescale_t);
        for (i, t) in times.windows(2).enumerate() {
            let time = mx::from_slice_f32(&[(t[0] * 1000.0) as f32], &[1]);
            let (v, u) = if c.slat_guidance > 0.0 && t[0] <= 0.5 {
                let (a, b) = model.latent.guided(
                    &latent,
                    &grid,
                    &time,
                    &condition,
                    uncondition
                        .as_ref()
                        .context("missing unconditional latent context")?,
                )?;
                (a, Some(b))
            } else {
                (
                    model.latent.forward(&latent, &grid, &time, &condition)?,
                    None,
                )
            };
            latent = update(&latent, &v, u.as_deref(), c.slat_guidance, t[1] - t[0]);
            mx::eval(&latent);
            eprintln!("sam3d_objects: latent {}/{}", i + 1, options.slat_steps);
        }
    }
    latent = mx::add(
        &mx::multiply(&latent, &mx::from_slice_f32(&c.slat_std, &[8])),
        &mx::from_slice_f32(&c.slat_mean, &[8]),
    );
    mx::eval(&latent);
    ensure!(
        mx::item_bool(&mx::all_all(&mx::isfinite(&latent))),
        "non-finite sparse latent"
    );
    let mut output = DenseOutput::default();
    for (name, value) in pose {
        output.tensors.insert(format!("pose.{name}"), value);
    }
    output
        .tensors
        .insert("coords".into(), mx::copy(&grid.coords));
    output.tensors.insert(
        "pointmap_conditioned".into(),
        mx::astype(
            &mx::from_slice_i32(&[i32::from(metadata.conditioned)], &[1]),
            mx::dtype::BOOL,
        ),
    );
    for kind in &options.formats {
        eprintln!("sam3d_objects: decode {kind}");
        if kind == "mesh" {
            let (features, mesh_grid) = model.mesh.forward(&latent, &grid)?;
            mx::eval(&features);
            output.meshes.push((
                kind.clone(),
                super::mesh::extract_mesh(&features, &mesh_grid)?,
            ));
        } else {
            let decoder = if kind == "gaussian" {
                &model.gaussian
            } else {
                &model.gaussian_4
            };
            output
                .gaussians
                .push((kind.clone(), decoder.forward(&latent, &grid)?));
        }
    }
    output.tensors.insert("latents".into(), latent);
    output.meta.extend(Map::from_iter([
        (String::from("seed"), json!(options.seed)),
        ("ss_steps".into(), json!(options.ss_steps)),
        ("slat_steps".into(), json!(options.slat_steps)),
        ("formats".into(), json!(options.formats)),
        ("pointmap_conditioned".into(), json!(metadata.conditioned)),
    ]));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3d_objects_options_preserve_defaults_and_reject_invalid_controls() {
        let c = Config::default();
        let o = Options::parse(&c, &Map::new()).unwrap();
        assert_eq!((o.seed, o.ss_steps, o.slat_steps), (42, 25, 25));
        assert_eq!(o.formats, ["gaussian", "mesh"]);
        assert!(o.estimate_depth);
        for value in [
            json!({"ss_steps":0}),
            json!({"slat_steps":true}),
            json!({"formats":[]}),
            json!({"formats":["mesh","mesh"]}),
            json!({"estimate_depth":"false"}),
        ] {
            assert!(Options::parse(&c, value.as_object().unwrap()).is_err());
        }
        assert_eq!(schedule(2, 3.0), [0.0, 0.25, 1.0]);
    }
}
