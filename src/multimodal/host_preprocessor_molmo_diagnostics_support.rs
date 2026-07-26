use std::fs;
use std::io::{self, Write};
use std::path::Path;

#[cfg(test)]
use image::DynamicImage;
use mlxcel_core::session::{PreparedPrefill, PreparedTensorDType};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::vision::processors::molmo::{MolmoImageTokens, MolmoProcessor, MolmoProcessorOutput};

pub(super) const PINNED_REVISION: &str = "5c04b3a418979597b1968e41414ad799c87533e8";
pub(super) const PINNED_CONFIG_SHA256: &str =
    "4b27cd3177a990224d4f31aa941bcf53b3bb99b90b097d0dd7084a99c423224c";
pub(super) const PINNED_PREPROCESSOR_SHA256: &str =
    "01a54ee26803f8a4088030ce2b6be93cf87d402fe8e8e4dbf56a2adbbb7ab3d7";
pub(super) const PINNED_PROCESSOR_SHA256: &str =
    "9c40772b47f74c061e4829611bb581e5498ac0f90ac0851874a907462a6a026e";
pub(super) const PINNED_IMAGE_SHA256: &str =
    "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261";
pub(super) const PINNED_PIXEL_VALUES_SHA256: &str =
    "0a1bef503b209de9b4454f636f03c55f4d84309cb7fd6bbf573750dd499799aa";
pub(super) const PINNED_IMAGE_MASKS_SHA256: &str =
    "5a0de017af320127aa8685a4671487960f2aa498b541d5445a0962410bf3b92a";
#[cfg(test)]
pub(super) const PINNED_IMAGE_TOKEN_IDS_SHA256: &str =
    "23a0878a6a2818f113460cc9a116f3d68f4d3d4b1a34b1f34dfe7805395562fa";
pub(super) const PINNED_IMAGE_INPUT_IDX_SHA256: &str =
    "243da1bc98ebb61f5d98696623b41ade1d643994db6fcfb764392452d3f9c1b4";
#[cfg(test)]
pub(super) const PINNED_GRID_SHA256: &str =
    "29c898799e5d25bb6fa02a097a3878d280419aeab84e720cab55d889c6821bd5";

#[derive(Debug, Clone, Copy)]
pub(super) struct Tolerance {
    atol: f64,
    rtol: f64,
}

pub(super) const EXACT: Tolerance = Tolerance {
    atol: 0.0,
    rtol: 0.0,
};
// The pinned host path retains F16 checkpoint rounding while the local-task
// StableHLO graph accumulates in F32. This is the existing #870 prepared-visual
// envelope, now applied at the first observable internal boundary.
pub(super) const VISION: Tolerance = Tolerance {
    atol: 0.25,
    rtol: 0.05,
};

pub(super) fn progress(stage: &str) {
    eprintln!("[molmo-v1-boundary] {stage}");
    io::stderr().flush().expect("flush diagnostic progress");
}

pub(super) fn compare_stage(
    stage: &str,
    observed: &[f32],
    reference: &[f32],
    tolerance: Tolerance,
    first_divergence: &mut Option<String>,
) {
    if observed.len() != reference.len() {
        let detail = format!("{stage}: length {} != {}", observed.len(), reference.len());
        first_divergence.get_or_insert(detail.clone());
        eprintln!("[molmo-v1-boundary] stage={stage} status=FAIL {detail}");
        return;
    }
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut failures = 0usize;
    let mut first_failure = None;
    for (index, (&observed, &reference)) in observed.iter().zip(reference).enumerate() {
        if !observed.is_finite() || !reference.is_finite() {
            failures += 1;
            first_failure.get_or_insert(index);
            continue;
        }
        let absolute = f64::from((observed - reference).abs());
        let relative = absolute / f64::from(reference.abs()).max(f64::MIN_POSITIVE);
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(relative);
        if absolute > tolerance.atol + tolerance.rtol * f64::from(reference.abs()) {
            failures += 1;
            first_failure.get_or_insert(index);
        }
    }
    let status = if failures == 0 { "PASS" } else { "FAIL" };
    eprintln!(
        "[molmo-v1-boundary] stage={stage} status={status} elements={} \
         atol={:.3e} rtol={:.3e} max_abs={max_abs:.6e} max_rel={max_rel:.6e} \
         failures={failures} first_failure={first_failure:?}",
        observed.len(),
        tolerance.atol,
        tolerance.rtol,
    );
    io::stderr().flush().expect("flush diagnostic stage");
    if failures != 0 {
        first_divergence.get_or_insert_with(|| {
            format!(
                "{stage} at flat index {}",
                first_failure.expect("failed comparison has an index")
            )
        });
    }
}

pub(super) fn compare_exact_i32(
    stage: &str,
    observed: &[i32],
    reference: &[i32],
    first_divergence: &mut Option<String>,
) {
    let first = observed
        .iter()
        .zip(reference)
        .position(|(observed, reference)| observed != reference);
    let failed = observed.len() != reference.len() || first.is_some();
    eprintln!(
        "[molmo-v1-boundary] stage={stage} status={} elements={} first_failure={first:?}",
        if failed { "FAIL" } else { "PASS" },
        observed.len()
    );
    if failed {
        first_divergence.get_or_insert_with(|| {
            format!(
                "{stage} at {}",
                first.map_or_else(
                    || format!("length {} != {}", observed.len(), reference.len()),
                    |index| format!("flat index {index}")
                )
            )
        });
    }
}

pub(super) fn sha256(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        )
    )
}

pub(super) fn assert_sha256(path: &Path, expected: &str, label: &str) {
    assert_eq!(
        sha256(path),
        expected,
        "{label} differs from the pinned #870 fixture: {}",
        path.display()
    );
}

pub(super) fn digest_f32(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub(super) fn digest_i32(values: &[i32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
pub(super) fn digest_grid(output: &MolmoProcessorOutput) -> String {
    let mut values = Vec::new();
    values.extend(output.pixel_values_shape);
    values.extend(output.image_masks_shape);
    values.push(output.image_input_idx_len);
    values.push(i32::try_from(output.image_token_ids.len()).expect("token count fits i32"));
    digest_i32(&values)
}

pub(super) fn pinned_revision(model: &Path) -> String {
    if let Ok(revision) = std::env::var("MLXCEL_MOLMO_REVISION") {
        return revision;
    }
    let metadata = model.join(".cache/huggingface/download/config.json.metadata");
    fs::read_to_string(&metadata)
        .unwrap_or_else(|error| {
            panic!(
                "read pinned revision from {} ({error}); set MLXCEL_MOLMO_REVISION",
                metadata.display()
            )
        })
        .lines()
        .next()
        .expect("Hugging Face metadata contains a revision")
        .to_string()
}

#[cfg(test)]
fn fixture_image() -> DynamicImage {
    image::open("tests/fixtures/test_image.png").expect("load pinned Molmo image fixture")
}

#[cfg(test)]
pub(super) fn pinned_processor_output() -> MolmoProcessorOutput {
    MolmoProcessor::new(
        12,
        Some((4, 4)),
        Some(14),
        Some((336, 336)),
        Some((12, 12)),
        None,
        None,
        MolmoImageTokens::default(),
    )
    .preprocess_image(&fixture_image())
}

pub(super) fn mlx_f32(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    let widened = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::try_array_to_raw_bytes(&widened)
        .expect("export MLX diagnostic tensor")
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

pub(super) fn prepared_f32(prepared: &PreparedPrefill) -> Vec<f32> {
    assert_eq!(prepared.embeddings.dtype, PreparedTensorDType::Float32);
    prepared
        .embeddings
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

pub(super) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}
