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

//! CLI handler for `mlxcel generate-image`.
//!
//! Runs text-to-image generation or reference-image editing with a Diffusers
//! style checkpoint and writes PNG files. Image models produce pixels rather
//! than tokens, so this is a separate surface from `generate`, like `detect`
//! is for boxes. Everything model-independent lives in `mlxcel::imagegen`.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use mlxcel::imagegen::run::{
    generate_to_files, parse_size, prepare_output_dir, random_seed, validate_request,
};
use mlxcel::imagegen::{
    FAMILIES, ImageFamily, ImageRequest, ImageTask, default_output_path, resolve_family,
};
use mlxcel::initialize_runtime_checked;

use crate::GenerateImageArgs;

/// `--task` values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TaskArg {
    /// Text-to-image.
    #[default]
    Generate,
    /// Edit one or more reference images.
    Edit,
}

impl From<TaskArg> for ImageTask {
    fn from(task: TaskArg) -> Self {
        match task {
            TaskArg::Generate => ImageTask::Generate,
            TaskArg::Edit => ImageTask::Edit,
        }
    }
}

/// Parse `--gen-kwargs`: a JSON object of model-specific knobs.
pub(crate) fn parse_gen_kwargs(raw: Option<&str>) -> Result<Map<String, Value>> {
    let Some(raw) = raw else {
        return Ok(Map::new());
    };
    match serde_json::from_str::<Value>(raw).context("--gen-kwargs is not valid JSON")? {
        Value::Object(map) => Ok(map),
        other => bail!(
            "--gen-kwargs must be a JSON object, got {}",
            match other {
                Value::Array(_) => "an array",
                Value::String(_) => "a string",
                Value::Number(_) => "a number",
                Value::Bool(_) => "a boolean",
                _ => "null",
            }
        ),
    }
}

/// Build the model-independent request from the CLI arguments.
///
/// Reads the reference images and picks the size, but loads no weights, so it
/// can run before the (slow) model load and fail fast.
pub(crate) fn build_request(args: &GenerateImageArgs, seed: u64) -> Result<ImageRequest> {
    let task = ImageTask::from(args.task);
    let mut references = Vec::with_capacity(args.image.len());
    for path in &args.image {
        if !path.exists() {
            bail!("Image file does not exist: {}", path.display());
        }
        references.push(
            image::open(path)
                .with_context(|| format!("cannot read reference image {}", path.display()))?,
        );
    }

    let (width, height) = match (args.size.as_deref(), task, references.first()) {
        (Some(size), _, _) => parse_size(size)?,
        (None, ImageTask::Edit, Some(first)) => {
            use image::GenericImageView;
            first.dimensions()
        }
        (None, _, _) => mlxcel::imagegen::pipeline::DEFAULT_IMAGE_SIZE,
    };

    let mut request = ImageRequest::generate(args.prompt.clone(), seed);
    request.negative_prompt = args.negative_prompt.clone();
    request.task = task;
    request.reference_images = references;
    request.width = width;
    request.height = height;
    request.steps = args.steps;
    request.guidance = args.guidance;
    request.n = args.n;
    request.extra = parse_gen_kwargs(args.gen_kwargs.as_deref())?;
    Ok(request)
}

/// Run the `mlxcel generate-image` subcommand.
pub(crate) fn run_generate_image(args: GenerateImageArgs) -> Result<()> {
    run_generate_image_with(args, FAMILIES)
}

/// [`run_generate_image`] against an explicit family table, so tests can drive
/// the whole command with a stub family.
pub(crate) fn run_generate_image_with(
    args: GenerateImageArgs,
    families: &[ImageFamily],
) -> Result<()> {
    if !args.model.exists() {
        return Err(anyhow!(
            "Model directory does not exist: {}",
            args.model.display()
        ));
    }
    let seed = args.seed.unwrap_or_else(random_seed);
    let request = build_request(&args, seed)?;
    // Everything below the model load needs the family's grid; these checks do not.
    if request.n < 1 || request.n > mlxcel::imagegen::pipeline::MAX_IMAGES_PER_REQUEST {
        bail!(
            "-n must be between 1 and {}, got {}",
            mlxcel::imagegen::pipeline::MAX_IMAGES_PER_REQUEST,
            request.n
        );
    }
    request.validate(1)?;

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(seed));
    // An unwritable output directory should cost milliseconds, not a model load.
    prepare_output_dir(&output)?;

    // Classify the checkpoint from its JSON metadata first: a wrong directory or
    // a family this build cannot run fails here, without touching the device.
    let family = resolve_family(&args.model, families)?;

    // Initialize the MLX runtime (selects GPU/CPU) before any forward pass.
    let _runtime = initialize_runtime_checked()?;

    let model = (family.load)(&args.model)?;
    validate_request(model.as_ref(), &request)?;

    println!("seed: {seed}");
    let paths = generate_to_files(model.as_ref(), &request, &output)?;
    for (index, path) in paths.iter().enumerate() {
        println!(
            "saved {} ({}x{}, seed {})",
            path.display(),
            request.width,
            request.height,
            request.seed_for(index as u32)
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "generate_image_tests.rs"]
mod tests;
