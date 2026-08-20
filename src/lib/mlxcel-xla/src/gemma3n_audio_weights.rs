//! Exact converted-checkpoint schema for the Gemma3n audio graph.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::Gemma3nXlaAudioConfig;

pub const GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT: usize = 277;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma3nAudioCheckpointDType {
    Bf16,
    U32,
}

impl Gemma3nAudioCheckpointDType {
    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::U32 => "U32",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma3nAudioCheckpointTensorSpec {
    pub name: String,
    pub dtype: Gemma3nAudioCheckpointDType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gemma3nAudioCheckpointError {
    Io(String),
    InvalidHeader(String),
    Missing(Vec<String>),
    Unexpected(Vec<String>),
    DType {
        name: String,
        actual: String,
        expected: &'static str,
    },
    Shape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
}

impl fmt::Display for Gemma3nAudioCheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) | Self::InvalidHeader(error) => f.write_str(error),
            Self::Missing(names) => {
                write!(f, "Gemma3n audio checkpoint is missing tensors: {names:?}")
            }
            Self::Unexpected(names) => {
                write!(
                    f,
                    "Gemma3n audio checkpoint has unexpected audio tensors: {names:?}"
                )
            }
            Self::DType {
                name,
                actual,
                expected,
            } => write!(
                f,
                "Gemma3n audio tensor {name} has dtype {actual}; expected {expected}"
            ),
            Self::Shape {
                name,
                actual,
                expected,
            } => write!(
                f,
                "Gemma3n audio tensor {name} has shape {actual:?}; expected {expected:?}"
            ),
        }
    }
}

impl std::error::Error for Gemma3nAudioCheckpointError {}

fn tensor(
    out: &mut Vec<Gemma3nAudioCheckpointTensorSpec>,
    name: impl Into<String>,
    dtype: Gemma3nAudioCheckpointDType,
    shape: impl Into<Vec<usize>>,
) {
    out.push(Gemma3nAudioCheckpointTensorSpec {
        name: name.into(),
        dtype,
        shape: shape.into(),
    });
}

fn dense(
    out: &mut Vec<Gemma3nAudioCheckpointTensorSpec>,
    name: impl Into<String>,
    shape: impl Into<Vec<usize>>,
) {
    tensor(out, name, Gemma3nAudioCheckpointDType::Bf16, shape);
}

fn feed_forward(out: &mut Vec<Gemma3nAudioCheckpointTensorSpec>, prefix: &str, hidden: usize) {
    dense(out, format!("{prefix}.pre_layer_norm.weight"), [hidden]);
    dense(
        out,
        format!("{prefix}.ffw_layer_1.weight"),
        [hidden * 4, hidden],
    );
    dense(
        out,
        format!("{prefix}.ffw_layer_2.weight"),
        [hidden, hidden * 4],
    );
    dense(out, format!("{prefix}.post_layer_norm.weight"), [hidden]);
}

fn affine_q4(
    out: &mut Vec<Gemma3nAudioCheckpointTensorSpec>,
    prefix: &str,
    rows: usize,
    columns: usize,
) {
    debug_assert!(columns.is_multiple_of(64));
    tensor(
        out,
        format!("{prefix}.weight"),
        Gemma3nAudioCheckpointDType::U32,
        [rows, columns / 8],
    );
    for suffix in ["scales", "biases"] {
        tensor(
            out,
            format!("{prefix}.{suffix}"),
            Gemma3nAudioCheckpointDType::Bf16,
            [rows, columns / 64],
        );
    }
}

/// The exact 277 raw tensors consumed from the pinned converted Q4 checkpoint.
pub fn gemma3n_audio_checkpoint_specs(
    config: &Gemma3nXlaAudioConfig,
    text_hidden: usize,
) -> Result<Vec<Gemma3nAudioCheckpointTensorSpec>, String> {
    config.validate()?;
    if text_hidden == 0 || !config.hidden_size.is_multiple_of(64) {
        return Err("Gemma3n audio Q4 dimensions must be positive multiples of 64".into());
    }
    let hidden = config.hidden_size;
    let channels = &config.sscp_conv_channel_size;
    let kernels = &config.sscp_conv_kernel_size;
    let mut out = Vec::with_capacity(GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT);
    dense(
        &mut out,
        "audio_tower.subsample_conv_projection.conv_0.conv.weight",
        [channels[0], kernels[0][0], kernels[0][1], 1],
    );
    dense(
        &mut out,
        "audio_tower.subsample_conv_projection.conv_0.norm.weight",
        [channels[0]],
    );
    dense(
        &mut out,
        "audio_tower.subsample_conv_projection.conv_1.conv.weight",
        [channels[1], kernels[1][0], kernels[1][1], channels[0]],
    );
    dense(
        &mut out,
        "audio_tower.subsample_conv_projection.conv_1.norm.weight",
        [channels[1]],
    );
    dense(
        &mut out,
        "audio_tower.subsample_conv_projection.input_proj_linear.weight",
        [hidden, config.output_frequency()? * channels[1]],
    );
    for layer in 0..config.conf_num_hidden_layers {
        let prefix = format!("audio_tower.conformer.{layer}");
        feed_forward(&mut out, &format!("{prefix}.ffw_layer_start"), hidden);
        let attention = format!("{prefix}.attention");
        dense(
            &mut out,
            format!("{attention}.pre_attn_norm.weight"),
            [hidden],
        );
        for projection in ["q_proj", "k_proj", "v_proj"] {
            dense(
                &mut out,
                format!("{attention}.attn.{projection}.weight"),
                [hidden, hidden],
            );
        }
        dense(
            &mut out,
            format!("{attention}.attn.per_dim_scale"),
            [config.head_dim()],
        );
        dense(
            &mut out,
            format!("{attention}.attn.relative_position_embedding.pos_proj.weight"),
            [hidden, hidden],
        );
        dense(
            &mut out,
            format!("{attention}.post.weight"),
            [hidden, hidden],
        );
        dense(&mut out, format!("{attention}.post_norm.weight"), [hidden]);
        let light = format!("{prefix}.lconv1d");
        dense(&mut out, format!("{light}.pre_layer_norm.weight"), [hidden]);
        dense(
            &mut out,
            format!("{light}.linear_start.weight"),
            [hidden * 2, hidden],
        );
        dense(
            &mut out,
            format!("{light}.depthwise_conv1d.weight"),
            [hidden, config.conf_conv_kernel_size, 1],
        );
        dense(&mut out, format!("{light}.conv_norm.weight"), [hidden]);
        dense(
            &mut out,
            format!("{light}.linear_end.weight"),
            [hidden, hidden],
        );
        feed_forward(&mut out, &format!("{prefix}.ffw_layer_end"), hidden);
        dense(&mut out, format!("{prefix}.norm.weight"), [hidden]);
    }
    affine_q4(&mut out, "embed_audio.embedding", config.vocab_size, hidden);
    affine_q4(
        &mut out,
        "embed_audio.embedding_projection",
        text_hidden,
        hidden,
    );
    dense(&mut out, "embed_audio.hard_embedding_norm.weight", [hidden]);
    dense(&mut out, "embed_audio.soft_embedding_norm.weight", [hidden]);
    if out.len() != GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT {
        return Err(format!(
            "Gemma3n audio schema generated {} tensors; expected {GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT}",
            out.len()
        ));
    }
    Ok(out)
}

#[derive(Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
}

fn safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>, Gemma3nAudioCheckpointError> {
    let mut paths = std::fs::read_dir(model_dir)
        .map_err(|error| {
            Gemma3nAudioCheckpointError::Io(format!("read {}: {error}", model_dir.display()))
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(Gemma3nAudioCheckpointError::Io(format!(
            "{} contains no safetensors files",
            model_dir.display()
        )));
    }
    Ok(paths)
}

fn read_header(path: &Path) -> Result<BTreeMap<String, HeaderTensor>, Gemma3nAudioCheckpointError> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        Gemma3nAudioCheckpointError::Io(format!("open {}: {error}", path.display()))
    })?;
    let mut size = [0_u8; 8];
    file.read_exact(&mut size).map_err(|error| {
        Gemma3nAudioCheckpointError::InvalidHeader(format!(
            "read {} header size: {error}",
            path.display()
        ))
    })?;
    let size = usize::try_from(u64::from_le_bytes(size)).map_err(|_| {
        Gemma3nAudioCheckpointError::InvalidHeader(format!(
            "{} header size exceeds this platform",
            path.display()
        ))
    })?;
    if size == 0 || size > 128 * 1024 * 1024 {
        return Err(Gemma3nAudioCheckpointError::InvalidHeader(format!(
            "{} has invalid safetensors header size {size}",
            path.display()
        )));
    }
    let mut bytes = vec![0_u8; size];
    file.seek(std::io::SeekFrom::Start(8))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|error| {
            Gemma3nAudioCheckpointError::InvalidHeader(format!(
                "read {} header: {error}",
                path.display()
            ))
        })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        Gemma3nAudioCheckpointError::InvalidHeader(format!(
            "parse {} header: {error}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        Gemma3nAudioCheckpointError::InvalidHeader(format!(
            "{} safetensors header must be an object",
            path.display()
        ))
    })?;
    object
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
        .map(|(name, value)| {
            serde_json::from_value(value.clone())
                .map(|metadata| (name.clone(), metadata))
                .map_err(|error| {
                    Gemma3nAudioCheckpointError::InvalidHeader(format!(
                        "{} tensor {name}: {error}",
                        path.display()
                    ))
                })
        })
        .collect()
}

pub fn validate_gemma3n_audio_checkpoint(
    model_dir: &Path,
    config: &Gemma3nXlaAudioConfig,
    text_hidden: usize,
) -> Result<(), Gemma3nAudioCheckpointError> {
    let mut actual = BTreeMap::new();
    for path in safetensor_files(model_dir)? {
        actual.extend(read_header(&path)?);
    }
    validate_gemma3n_audio_tensor_map(&actual, config, text_hidden)
}

fn validate_gemma3n_audio_tensor_map(
    actual: &BTreeMap<String, HeaderTensor>,
    config: &Gemma3nXlaAudioConfig,
    text_hidden: usize,
) -> Result<(), Gemma3nAudioCheckpointError> {
    let specs = gemma3n_audio_checkpoint_specs(config, text_hidden)
        .map_err(Gemma3nAudioCheckpointError::InvalidHeader)?;
    let expected_names = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing = expected_names
        .iter()
        .filter(|name| !actual.contains_key(**name))
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Gemma3nAudioCheckpointError::Missing(missing));
    }
    let unexpected = actual
        .keys()
        .filter(|name| {
            (name.starts_with("audio_tower.") || name.starts_with("embed_audio."))
                && !expected_names.contains(name.as_str())
        })
        .cloned()
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(Gemma3nAudioCheckpointError::Unexpected(unexpected));
    }
    for spec in specs {
        let metadata = &actual[&spec.name];
        if metadata.dtype != spec.dtype.name() {
            return Err(Gemma3nAudioCheckpointError::DType {
                name: spec.name,
                actual: metadata.dtype.clone(),
                expected: spec.dtype.name(),
            });
        }
        if metadata.shape != spec.shape {
            return Err(Gemma3nAudioCheckpointError::Shape {
                name: spec.name,
                actual: metadata.shape.clone(),
                expected: spec.shape,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_from_specs(
        specs: &[Gemma3nAudioCheckpointTensorSpec],
    ) -> BTreeMap<String, HeaderTensor> {
        specs
            .iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    HeaderTensor {
                        dtype: spec.dtype.name().into(),
                        shape: spec.shape.clone(),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn pinned_q4_schema_has_exact_order_count_and_boundaries() {
        let specs =
            gemma3n_audio_checkpoint_specs(&Gemma3nXlaAudioConfig::default(), 2_048).unwrap();
        assert_eq!(specs.len(), GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT);
        assert_eq!(
            specs[0].name,
            "audio_tower.subsample_conv_projection.conv_0.conv.weight"
        );
        assert_eq!(specs[0].shape, [128, 3, 3, 1]);
        assert_eq!(specs[268].name, "audio_tower.conformer.11.norm.weight");
        assert_eq!(specs[269].name, "embed_audio.embedding.weight");
        assert_eq!(specs[269].shape, [128, 192]);
        assert_eq!(
            specs.last().unwrap().name,
            "embed_audio.soft_embedding_norm.weight"
        );
    }

    #[test]
    fn validation_rejects_missing_extra_dtype_and_shape() {
        let config = Gemma3nXlaAudioConfig::default();
        let specs = gemma3n_audio_checkpoint_specs(&config, 2_048).unwrap();
        let mut actual = map_from_specs(&specs);
        actual.remove(&specs[7].name);
        assert!(matches!(
            validate_gemma3n_audio_tensor_map(&actual, &config, 2_048),
            Err(Gemma3nAudioCheckpointError::Missing(_))
        ));
        let mut actual = map_from_specs(&specs);
        actual.insert(
            "audio_tower.unqualified.weight".into(),
            HeaderTensor {
                dtype: "BF16".into(),
                shape: vec![1],
            },
        );
        assert!(matches!(
            validate_gemma3n_audio_tensor_map(&actual, &config, 2_048),
            Err(Gemma3nAudioCheckpointError::Unexpected(_))
        ));
        let mut actual = map_from_specs(&specs);
        actual.get_mut(&specs[0].name).unwrap().dtype = "F32".into();
        assert!(matches!(
            validate_gemma3n_audio_tensor_map(&actual, &config, 2_048),
            Err(Gemma3nAudioCheckpointError::DType { .. })
        ));
        let mut actual = map_from_specs(&specs);
        actual.get_mut(&specs[0].name).unwrap().shape[0] = 127;
        assert!(matches!(
            validate_gemma3n_audio_tensor_map(&actual, &config, 2_048),
            Err(Gemma3nAudioCheckpointError::Shape { .. })
        ));
    }

    #[test]
    #[ignore = "requires GEMMA3N_MODEL_DIR pointing at a converted Q4 checkpoint"]
    fn real_checkpoint_has_exact_audio_schema() {
        let model_dir =
            PathBuf::from(std::env::var_os("GEMMA3N_MODEL_DIR").expect("model directory"));
        let config = Gemma3nXlaAudioConfig::from_model_dir(&model_dir)
            .unwrap()
            .unwrap();
        validate_gemma3n_audio_checkpoint(&model_dir, &config, 2_048).unwrap();
    }
}
