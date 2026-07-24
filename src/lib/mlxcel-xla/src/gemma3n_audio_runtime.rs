//! Conditional two-artifact IREE runtime for Gemma3n audio.
//!
//! The encoder and language merge are separate resident-weight modules. This
//! keeps the compiler at natural tensor boundaries while ensuring every model
//! operation after host mel extraction still executes through IREE.

#[cfg(feature = "diagnostics")]
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    AuxiliaryWeightStorage, IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, ensure_qualified_auxiliary_artifact};
use crate::emitter::{
    Gemma3nAudioGraphWeightSpec, Gemma3nConfig, Precision, emit_gemma3n_audio_encode,
    emit_gemma3n_audio_merge_ple, gemma3n_audio_encoder_weights, gemma3n_audio_merge_weights,
};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::weights::{bf16_to_f32, dequantize_affine_bf16_sequential, f16_to_f32, f32_le_to_f32};
use crate::{
    GEMMA3N_AUDIO_MODALITY_FAMILY, GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nAudioInput,
    Gemma3nAudioPreparedPrefill, Gemma3nDensePle, Gemma3nPreparedPrefill, Gemma3nXlaAudioConfig,
    validate_gemma3n_audio_checkpoint, validate_gemma3n_audio_row_indices,
};
use mlxcel_core::session::{
    OwnedTensor, PreparedAttentionBias, PreparedModality, PreparedPositions, PreparedPrefill,
    PreparedTensorDType,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma3nAudioGraphOutput {
    embeddings: Vec<f32>,
    dense_ple: Vec<f32>,
    projected_lengths: Vec<usize>,
    #[cfg(feature = "diagnostics")]
    projected_audio: Vec<f32>,
    #[cfg(feature = "diagnostics")]
    hard_audio: Vec<f32>,
    #[cfg(feature = "diagnostics")]
    diagnostics: BTreeMap<String, Vec<f32>>,
    context_capacity: usize,
    hidden_size: usize,
    layers: usize,
    hidden_per_layer: usize,
}

impl Gemma3nAudioGraphOutput {
    #[must_use]
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    #[must_use]
    pub fn dense_ple(&self) -> &[f32] {
        &self.dense_ple
    }

    #[must_use]
    pub fn projected_lengths(&self) -> &[usize] {
        &self.projected_lengths
    }

    #[cfg(feature = "diagnostics")]
    #[must_use]
    pub fn projected_audio(&self) -> &[f32] {
        &self.projected_audio
    }

    #[cfg(feature = "diagnostics")]
    #[must_use]
    pub fn hard_audio(&self) -> &[f32] {
        &self.hard_audio
    }

    #[cfg(feature = "diagnostics")]
    #[must_use]
    pub fn diagnostic_stage(&self, name: &str) -> Option<&[f32]> {
        self.diagnostics.get(name).map(Vec::as_slice)
    }

    #[must_use]
    pub fn embeddings_shape(&self) -> [usize; 2] {
        [self.context_capacity, self.hidden_size]
    }

    #[must_use]
    pub fn dense_ple_shape(&self) -> [usize; 3] {
        [self.context_capacity, self.layers, self.hidden_per_layer]
    }

    pub fn into_parts(self) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
        (self.embeddings, self.dense_ple, self.projected_lengths)
    }

    /// Move a verified split-graph result into the existing #876 prepared
    /// embeddings-plus-dense-PLE contract.
    pub fn into_prepared(
        self,
        token_ids: Vec<i32>,
        audio_token_id: i32,
        frame_bucket: usize,
    ) -> Result<Gemma3nAudioPreparedPrefill, String> {
        let sequence_len = token_ids.len();
        if sequence_len == 0 || sequence_len > self.context_capacity {
            return Err(format!(
                "Gemma3n audio prepared length {sequence_len} is outside 1..={}",
                self.context_capacity
            ));
        }
        let embedding_elements = sequence_len
            .checked_mul(self.hidden_size)
            .ok_or_else(|| "Gemma3n audio embedding length overflows".to_string())?;
        let embeddings = OwnedTensor::new(
            self.embeddings[..embedding_elements]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            PreparedTensorDType::Float32,
            vec![1, sequence_len, self.hidden_size],
        )
        .map_err(|error| error.to_string())?;
        let attention_bias = OwnedTensor::new(
            vec![0; sequence_len * std::mem::size_of::<f32>()],
            PreparedTensorDType::Float32,
            vec![1, 1, 1, sequence_len],
        )
        .map_err(|error| error.to_string())?;
        let clips = self.projected_lengths.len();
        let prepared = PreparedPrefill::new(
            token_ids,
            embeddings,
            PreparedPositions::Sequential {
                start: 0,
                length: sequence_len,
            },
            PreparedAttentionBias {
                tensor: attention_bias,
                causal: true,
            },
            vec![PreparedModality {
                family: GEMMA3N_AUDIO_MODALITY_FAMILY.to_string(),
                item_count: clips,
                token_count: clips * GEMMA3N_AUDIO_SOFT_TOKENS,
            }],
        )
        .map_err(|error| error.to_string())?;
        let dense_ple = Gemma3nDensePle::new(
            self.dense_ple,
            self.context_capacity,
            self.layers,
            self.hidden_per_layer,
        )
        .map_err(|error| error.to_string())?;
        let request =
            Gemma3nPreparedPrefill::new(prepared, dense_ple).map_err(|error| error.to_string())?;
        Gemma3nAudioPreparedPrefill::new(
            request,
            audio_token_id,
            self.projected_lengths,
            frame_bucket,
        )
    }
}

pub struct Gemma3nAudioIreeRuntime {
    encode: IreeAuxiliaryModule,
    merge_ple: IreeAuxiliaryModule,
    audio: Gemma3nXlaAudioConfig,
    frame_bucket: usize,
    clips: usize,
    context_capacity: usize,
    hidden_size: usize,
    layers: usize,
    hidden_per_layer: usize,
    language_fingerprint: u64,
    #[cfg(feature = "diagnostics")]
    diagnostic_shapes: Vec<(String, Vec<usize>)>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn generation_identity(compiler: &Path, flags: &[&str], mlir: &str) -> Result<String, String> {
    let version = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    if !version.status.success() {
        return Err(format!(
            "{} --version failed: {}",
            compiler.display(),
            String::from_utf8_lossy(&version.stderr)
        ));
    }
    generation_identity_from_version(
        compiler,
        flags,
        mlir,
        String::from_utf8_lossy(&version.stdout).trim(),
    )
}

fn generation_identity_from_version(
    compiler: &Path,
    flags: &[&str],
    mlir: &str,
    version: &str,
) -> Result<String, String> {
    Ok(format!(
        "compiler={};compiler_sha256={};version={version};flags={flags:?};mlir_sha256={}",
        compiler.display(),
        sha256_file(compiler)?,
        sha256_hex(mlir.as_bytes())
    ))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex_bytes(&digest.finalize()))
}

fn checkpoint_alias(name: &str) -> Option<String> {
    name.strip_prefix("model.language_model.")
        .map(|suffix| format!("language_model.model.{suffix}"))
}

fn safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = std::fs::read_dir(model_dir)
        .map_err(|error| format!("read {}: {error}", model_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".safetensors"))
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "{} contains no safetensors files",
            model_dir.display()
        ));
    }
    Ok(files)
}

fn decode_weight(
    tensors: &SafeTensors<'_>,
    checkpoint_name: &str,
    spec: &Gemma3nAudioGraphWeightSpec,
    quant_bits: usize,
    quant_group: usize,
) -> Result<Vec<f32>, String> {
    let tensor = tensors
        .tensor(checkpoint_name)
        .map_err(|error| format!("read {checkpoint_name}: {error}"))?;
    let (values, shape) = match tensor.dtype() {
        Dtype::BF16 => (bf16_to_f32(tensor.data()), tensor.shape().to_vec()),
        Dtype::F16 => (f16_to_f32(tensor.data()), tensor.shape().to_vec()),
        Dtype::F32 => (f32_le_to_f32(tensor.data()), tensor.shape().to_vec()),
        Dtype::U32 => {
            let prefix = checkpoint_name
                .strip_suffix(".weight")
                .ok_or_else(|| format!("quantized tensor {checkpoint_name} is not a weight"))?;
            let scales_name = format!("{prefix}.scales");
            let biases_name = format!("{prefix}.biases");
            let scales = tensors
                .tensor(&scales_name)
                .map_err(|error| format!("read {scales_name}: {error}"))?;
            let biases = tensors
                .tensor(&biases_name)
                .map_err(|error| format!("read {biases_name}: {error}"))?;
            if scales.dtype() != Dtype::BF16 || biases.dtype() != Dtype::BF16 {
                return Err(format!(
                    "{prefix} affine metadata must be BF16, got {:?}/{:?}",
                    scales.dtype(),
                    biases.dtype()
                ));
            }
            let packed = tensor.shape();
            if packed.len() != 2 {
                return Err(format!(
                    "quantized tensor {checkpoint_name} rank {} is not 2",
                    packed.len()
                ));
            }
            let logical_columns = packed[1]
                .checked_mul(32 / quant_bits)
                .ok_or_else(|| format!("{checkpoint_name} logical width overflows"))?;
            let values = dequantize_affine_bf16_sequential(
                tensor.data(),
                scales.data(),
                biases.data(),
                packed[0],
                packed[1],
                quant_bits,
                quant_group,
            )
            .map_err(|error| format!("dequantize {checkpoint_name}: {error}"))?;
            (values, vec![packed[0], logical_columns])
        }
        dtype => {
            return Err(format!(
                "weight {checkpoint_name} dtype {dtype:?} is not BF16/F16/F32/U32"
            ));
        }
    };
    if shape != spec.shape {
        return Err(format!(
            "weight {} has logical shape {shape:?}, expected {:?}",
            spec.name, spec.shape
        ));
    }
    validate_finite_values(&format!("Gemma3n auxiliary weight {}", spec.name), &values)?;
    Ok(values)
}

fn validate_finite_values(label: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} contains non-finite value {value} at flat index {index}"
        ));
    }
    Ok(())
}

fn load_weights(
    model_dir: &Path,
    specs: &[Gemma3nAudioGraphWeightSpec],
    text: &Gemma3nConfig,
) -> Result<Vec<AuxiliaryWeight>, String> {
    let quant = text
        .quantization
        .ok_or_else(|| "Gemma3n audio runtime requires the pinned Q4 config".to_string())?;
    let mut loaded = (0..specs.len()).map(|_| None).collect::<Vec<_>>();
    for path in safetensor_files(model_dir)? {
        let file =
            File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
        // Safety: the file remains open and the mapping is read-only for this scope.
        let mapping = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", path.display()))?;
        let tensors = SafeTensors::deserialize(&mapping)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        for (index, spec) in specs.iter().enumerate() {
            let canonical = tensors.tensor(&spec.name).is_ok();
            let alias = checkpoint_alias(&spec.name);
            let repackaged = alias
                .as_deref()
                .is_some_and(|name| tensors.tensor(name).is_ok());
            let checkpoint_name = match (canonical, repackaged) {
                (false, false) => continue,
                (true, false) => spec.name.clone(),
                (false, true) => alias.expect("repackaged presence requires an alias"),
                (true, true) => {
                    return Err(format!(
                        "{} contains both canonical and repackaged {}",
                        path.display(),
                        spec.name
                    ));
                }
            };
            if loaded[index].is_some() {
                return Err(format!(
                    "Gemma3n auxiliary weight {} occurs in multiple shards",
                    spec.name
                ));
            }
            let values = decode_weight(
                &tensors,
                &checkpoint_name,
                spec,
                quant.bits,
                quant.group_size,
            )?;
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                storage: AuxiliaryWeightStorage::Float32(values),
                dtype: AuxiliaryWeightDType::Float32,
                shape: spec.shape.clone(),
            });
        }
    }
    loaded
        .into_iter()
        .zip(specs)
        .map(|(weight, spec)| {
            weight.ok_or_else(|| format!("Gemma3n auxiliary weight {} is missing", spec.name))
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // Safety: f32 is plain data and the returned view cannot outlive `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // Safety: f32 is plain data, the byte view covers exactly the same
    // allocation, and the exclusive borrow prevents typed access until return.
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}

fn i32_bytes(values: &[i32]) -> &[u8] {
    // Safety: i32 is plain data and the returned view cannot outlive `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn i32_bytes_mut(values: &mut [i32]) -> &mut [u8] {
    // Safety: see `f32_bytes_mut`; the same invariant holds for i32.
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}

fn load_timing_enabled() -> bool {
    std::env::var_os("MLXCEL_XLA_LOAD_TIMING").is_some()
}

fn current_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

fn report_load_timing(label: &str, started: std::time::Instant) {
    if load_timing_enabled() {
        eprintln!(
            "mlxcel-xla-load: phase={label} elapsed_ms={} rss_kib={}",
            started.elapsed().as_millis(),
            current_rss_kib()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unavailable".to_string())
        );
    }
}

fn report_auxiliary_weights(label: &str, started: std::time::Instant, weights: &[AuxiliaryWeight]) {
    if load_timing_enabled() {
        let f32_bytes = weights
            .iter()
            .filter(|weight| matches!(&weight.storage, AuxiliaryWeightStorage::Float32(_)))
            .map(|weight| weight.storage.byte_len())
            .sum::<usize>();
        let raw_bytes = weights
            .iter()
            .filter(|weight| matches!(&weight.storage, AuxiliaryWeightStorage::Bytes(_)))
            .map(|weight| weight.storage.byte_len())
            .sum::<usize>();
        eprintln!(
            "mlxcel-xla-load: phase={label} elapsed_ms={} rss_kib={} \
             buffers={} f32_bytes={f32_bytes} raw_bytes={raw_bytes}",
            started.elapsed().as_millis(),
            current_rss_kib()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
            weights.len()
        );
    }
}

impl Gemma3nAudioIreeRuntime {
    #[cfg(feature = "diagnostics")]
    pub fn load_audio_only_diagnostic(
        model_dir: &Path,
        device: &str,
        context_capacity: usize,
        frame_bucket: usize,
        clips: usize,
    ) -> Result<Self, String> {
        const AUDIO_ONLY_DIAGNOSTIC_IDENTITY: u64 = 0x6175_6469_6f2d_6469;
        Self::load(
            model_dir,
            device,
            context_capacity,
            frame_bucket,
            clips,
            AUDIO_ONLY_DIAGNOSTIC_IDENTITY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model_dir: &Path,
        device: &str,
        context_capacity: usize,
        frame_bucket: usize,
        clips: usize,
        language_fingerprint: u64,
    ) -> Result<Self, String> {
        let load_started = std::time::Instant::now();
        if language_fingerprint == 0 {
            return Err("Gemma3n audio requires a verified #876 language bundle".to_string());
        }
        let text = Gemma3nConfig::from_json(model_dir)?.with_context_capacity(context_capacity)?;
        let audio = Gemma3nXlaAudioConfig::from_model_dir(model_dir)?
            .ok_or_else(|| "checkpoint has no compatible Gemma3n audio config".to_string())?;
        validate_gemma3n_audio_checkpoint(model_dir, &audio, text.hidden)
            .map_err(|error| error.to_string())?;
        report_load_timing("audio-config-and-schema-validation", load_started);
        let emit_started = std::time::Instant::now();
        let (encode_mlir, encode_layout) =
            emit_gemma3n_audio_encode(&text, &audio, frame_bucket, clips, Precision::Bf16)?;
        let (merge_mlir, _) = emit_gemma3n_audio_merge_ple(&text, &audio, clips, Precision::Bf16)?;
        report_load_timing("audio-stablehlo-emission", emit_started);
        let compiler = iree_compile_bin()?;
        let mut flags = target_flags(device)?.to_vec();
        if device == "cuda" {
            flags.push("--iree-cuda-target=sm_80");
        }
        let cache = std::env::temp_dir().join("mlxcel-xla-gemma3n-audio");
        std::fs::create_dir_all(&cache)
            .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
        let encode_tag = format!("audio-encode-f{frame_bucket}-b{clips}");
        let merge_tag = format!("audio-merge-ple-c{context_capacity}-b{clips}");
        let encode_vmfb = cached_vmfb_path(
            &compiler,
            &encode_mlir,
            &flags,
            &cache,
            &encode_tag,
            context_capacity,
        );
        let merge_vmfb = cached_vmfb_path(
            &compiler,
            &merge_mlir,
            &flags,
            &cache,
            &merge_tag,
            context_capacity,
        );
        let identity = audio.artifact_identity(
            frame_bucket,
            clips,
            text.hidden,
            text.n_layers,
            text.hidden_per_layer_input,
        )?;
        let encode_contract = AuxiliaryArtifactContract::new(
            "audio_encode.main",
            format!("encode:{identity}:language={language_fingerprint:016x}"),
            generation_identity(&compiler, &flags, &encode_mlir)?,
        )?;
        let merge_contract = AuxiliaryArtifactContract::new(
            "audio_merge_ple.main",
            format!(
                "merge:{identity}:context={context_capacity}:text={}:language={language_fingerprint:016x}",
                text.compatibility_fingerprint()
            ),
            generation_identity(&compiler, &flags, &merge_mlir)?,
        )?;
        let encode_weights_started = std::time::Instant::now();
        let encode_weights = load_weights(
            model_dir,
            &gemma3n_audio_encoder_weights(&audio, &text)?,
            &text,
        )?;
        report_auxiliary_weights(
            "audio-encoder-weights",
            encode_weights_started,
            &encode_weights,
        );
        let encode_compile_started = std::time::Instant::now();
        ensure_qualified_auxiliary_artifact(
            &encode_vmfb,
            &encode_contract,
            &encode_weights,
            |temporary| {
                compile_one_to(
                    &compiler,
                    &encode_mlir,
                    &flags,
                    &cache,
                    &encode_tag,
                    context_capacity,
                    temporary,
                )
            },
        )?;
        report_load_timing("audio-encoder-vmfb-qualified", encode_compile_started);
        let encode_module_started = std::time::Instant::now();
        let encode =
            IreeAuxiliaryModule::load(device, &encode_vmfb, &encode_contract, encode_weights)?;
        report_load_timing("audio-encoder-vmfb-loaded", encode_module_started);
        let merge_weights_started = std::time::Instant::now();
        let merge_weights = load_weights(model_dir, &gemma3n_audio_merge_weights(&text), &text)?;
        report_auxiliary_weights("audio-merge-weights", merge_weights_started, &merge_weights);
        let merge_compile_started = std::time::Instant::now();
        ensure_qualified_auxiliary_artifact(
            &merge_vmfb,
            &merge_contract,
            &merge_weights,
            |temporary| {
                compile_one_to(
                    &compiler,
                    &merge_mlir,
                    &flags,
                    &cache,
                    &merge_tag,
                    context_capacity,
                    temporary,
                )
            },
        )?;
        report_load_timing("audio-merge-vmfb-qualified", merge_compile_started);
        let merge_module_started = std::time::Instant::now();
        let merge_ple =
            IreeAuxiliaryModule::load(device, &merge_vmfb, &merge_contract, merge_weights)?;
        report_load_timing("audio-merge-vmfb-loaded", merge_module_started);
        report_load_timing("audio-total", load_started);
        #[cfg(feature = "diagnostics")]
        let diagnostic_shapes = [
            "sscp_conv_0",
            "encoded_reduced",
            "soft_norm",
            "soft_linear",
            "soft_post_norm",
            "hard_embedding",
            "hard_norm",
            "hard_linear",
            "hard_post_norm",
        ]
        .into_iter()
        .map(|name| {
            encode_layout
                .stages
                .iter()
                .find(|stage| stage.name == name)
                .map(|stage| (name.to_string(), stage.shape.clone()))
                .ok_or_else(|| format!("Gemma3n audio diagnostic stage {name} is missing"))
        })
        .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "diagnostics"))]
        let _ = encode_layout;
        Ok(Self {
            encode,
            merge_ple,
            audio,
            frame_bucket,
            clips,
            context_capacity,
            hidden_size: text.hidden,
            layers: text.n_layers,
            hidden_per_layer: text.hidden_per_layer_input,
            language_fingerprint,
            #[cfg(feature = "diagnostics")]
            diagnostic_shapes,
        })
    }

    #[must_use]
    pub fn has_capability(&self) -> bool {
        self.language_fingerprint != 0
            && self.encode.fingerprint() != 0
            && self.merge_ple.fingerprint() != 0
    }

    pub fn invoke(
        &mut self,
        input: &Gemma3nAudioInput,
        token_ids: &[i32],
        audio_row_indices: &[i32],
    ) -> Result<Gemma3nAudioGraphOutput, String> {
        let invoke_started = std::time::Instant::now();
        if input.frame_bucket() != self.frame_bucket || input.clips() != self.clips {
            return Err(format!(
                "Gemma3n audio runtime expects bucket/clips {}/{}, got {}/{}",
                self.frame_bucket,
                self.clips,
                input.frame_bucket(),
                input.clips()
            ));
        }
        if token_ids.is_empty() || token_ids.len() > self.context_capacity {
            return Err(format!(
                "Gemma3n audio expanded length {} is outside 1..={}",
                token_ids.len(),
                self.context_capacity
            ));
        }
        validate_gemma3n_audio_row_indices(
            token_ids,
            self.audio.vocab_offset + 1,
            self.clips,
            audio_row_indices,
        )
        .map_err(|error| error.to_string())?;

        let projected_rows = self.clips * GEMMA3N_AUDIO_SOFT_TOKENS;
        let mut projected = vec![0.0f32; projected_rows * self.hidden_size];
        let mut hard = vec![0.0f32; self.audio.vocab_size * self.hidden_size];
        let mut lengths = vec![0i32; self.clips];
        let encode_started = std::time::Instant::now();
        let encode_inputs = [
            AuxiliaryInput {
                bytes: f32_bytes(input.mel()),
                dtype: AuxiliaryTensorDType::Float32,
                shape: &[self.clips, self.frame_bucket, self.audio.input_feat_size],
            },
            AuxiliaryInput {
                bytes: input.valid_mask(),
                dtype: AuxiliaryTensorDType::Bool,
                shape: &[self.clips, self.frame_bucket],
            },
        ];
        #[cfg(not(feature = "diagnostics"))]
        self.encode.invoke(
            &encode_inputs,
            &mut [
                AuxiliaryOutput {
                    bytes: f32_bytes_mut(&mut projected),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[projected_rows, self.hidden_size],
                },
                AuxiliaryOutput {
                    bytes: f32_bytes_mut(&mut hard),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[self.audio.vocab_size, self.hidden_size],
                },
                AuxiliaryOutput {
                    bytes: i32_bytes_mut(&mut lengths),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[self.clips],
                },
            ],
        )?;
        #[cfg(feature = "diagnostics")]
        let diagnostics = {
            let diagnostic_shapes = self.diagnostic_shapes.clone();
            let mut diagnostic_values = diagnostic_shapes
                .iter()
                .map(|(_, shape)| vec![0.0f32; shape.iter().product()])
                .collect::<Vec<_>>();
            let projected_shape = [projected_rows, self.hidden_size];
            let hard_shape = [self.audio.vocab_size, self.hidden_size];
            let lengths_shape = [self.clips];
            let mut outputs = vec![
                AuxiliaryOutput {
                    bytes: f32_bytes_mut(&mut projected),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &projected_shape,
                },
                AuxiliaryOutput {
                    bytes: f32_bytes_mut(&mut hard),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &hard_shape,
                },
                AuxiliaryOutput {
                    bytes: i32_bytes_mut(&mut lengths),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &lengths_shape,
                },
            ];
            for (values, (_, shape)) in diagnostic_values.iter_mut().zip(diagnostic_shapes.iter()) {
                outputs.push(AuxiliaryOutput {
                    bytes: f32_bytes_mut(values),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape,
                });
            }
            self.encode.invoke(&encode_inputs, &mut outputs)?;
            drop(outputs);
            diagnostic_shapes
                .into_iter()
                .map(|(name, _)| name)
                .zip(diagnostic_values)
                .collect::<BTreeMap<_, _>>()
        };
        report_load_timing("audio-encoder-invoke", encode_started);
        let projected_lengths = lengths
            .into_iter()
            .enumerate()
            .map(|(clip, length)| {
                usize::try_from(length)
                    .ok()
                    .filter(|length| (1..=GEMMA3N_AUDIO_SOFT_TOKENS).contains(length))
                    .ok_or_else(|| {
                        format!("audio.encode returned invalid length {length} for clip {clip}")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let real_len = i32::try_from(token_ids.len())
            .map_err(|_| "Gemma3n audio expanded length does not fit i32".to_string())?;
        let mut padded_tokens = token_ids.to_vec();
        padded_tokens.resize(self.context_capacity, 0);
        let mut padded_rows = audio_row_indices.to_vec();
        padded_rows.resize(self.context_capacity, -1);
        let mut embeddings = vec![0.0f32; self.context_capacity * self.hidden_size];
        let mut dense_ple =
            vec![0.0f32; self.context_capacity * self.layers * self.hidden_per_layer];
        let merge_started = std::time::Instant::now();
        self.merge_ple.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_bytes(&projected),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[projected_rows, self.hidden_size],
                },
                AuxiliaryInput {
                    bytes: f32_bytes(&hard),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[self.audio.vocab_size, self.hidden_size],
                },
                AuxiliaryInput {
                    bytes: i32_bytes(&padded_tokens),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[self.context_capacity],
                },
                AuxiliaryInput {
                    bytes: i32_bytes(&padded_rows),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[self.context_capacity],
                },
                AuxiliaryInput {
                    bytes: i32_bytes(std::slice::from_ref(&real_len)),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[],
                },
            ],
            &mut [
                AuxiliaryOutput {
                    bytes: f32_bytes_mut(&mut embeddings),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[self.context_capacity, self.hidden_size],
                },
                AuxiliaryOutput {
                    bytes: f32_bytes_mut(&mut dense_ple),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[self.context_capacity, self.layers, self.hidden_per_layer],
                },
            ],
        )?;
        report_load_timing("audio-merge-invoke", merge_started);
        report_load_timing("audio-invoke-total", invoke_started);
        Ok(Gemma3nAudioGraphOutput {
            embeddings,
            dense_ple,
            projected_lengths,
            #[cfg(feature = "diagnostics")]
            projected_audio: projected,
            #[cfg(feature = "diagnostics")]
            hard_audio: hard,
            #[cfg(feature = "diagnostics")]
            diagnostics,
            context_capacity: self.context_capacity,
            hidden_size: self.hidden_size,
            layers: self.layers,
            hidden_per_layer: self.hidden_per_layer,
        })
    }

    /// Run both split audio modules and move their output directly into the
    /// request-scoped #876 prefill owner.
    pub fn invoke_prepared(
        &mut self,
        input: &Gemma3nAudioInput,
        token_ids: Vec<i32>,
        audio_token_id: i32,
    ) -> Result<Gemma3nAudioPreparedPrefill, String> {
        let audio_row_indices = crate::gemma3n_audio_rows::build_gemma3n_audio_row_indices(
            &token_ids,
            audio_token_id,
            input.clips(),
        )
        .map_err(|error| error.to_string())?;
        let output = self.invoke(input, &token_ids, &audio_row_indices)?;
        output.into_prepared(token_ids, audio_token_id, input.frame_bucket())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux_manifest::auxiliary_manifest_path;

    fn temp_path(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mlxcel-xla-audio-{tag}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn output_moves_into_request_scoped_dense_ple_owner() {
        let context_capacity = GEMMA3N_AUDIO_SOFT_TOKENS + 2;
        let mut token_ids = vec![7];
        token_ids.extend(std::iter::repeat_n(262_273, GEMMA3N_AUDIO_SOFT_TOKENS));
        token_ids.push(9);
        let output = Gemma3nAudioGraphOutput {
            embeddings: vec![0.25; context_capacity * 2],
            dense_ple: vec![0.5; context_capacity * 2],
            projected_lengths: vec![GEMMA3N_AUDIO_SOFT_TOKENS],
            #[cfg(feature = "diagnostics")]
            projected_audio: Vec::new(),
            #[cfg(feature = "diagnostics")]
            hard_audio: Vec::new(),
            #[cfg(feature = "diagnostics")]
            diagnostics: BTreeMap::new(),
            context_capacity,
            hidden_size: 2,
            layers: 1,
            hidden_per_layer: 2,
        };
        let prepared = output
            .into_prepared(token_ids, 262_273, 2_048)
            .expect("valid audio prepared prefill");
        assert_eq!(prepared.projected_lengths(), [GEMMA3N_AUDIO_SOFT_TOKENS]);
        assert_eq!(prepared.placeholder_starts(), [1]);
        assert_eq!(
            prepared.request().dense_ple().shape(),
            [context_capacity, 1, 2]
        );
    }

    #[test]
    fn zero_language_bundle_identity_fails_before_checkpoint_access() {
        let error = Gemma3nAudioIreeRuntime::load(
            Path::new("/path/that/must/not/be/read"),
            "local-task",
            256,
            2_048,
            1,
            0,
        )
        .err()
        .expect("zero identity must fail");
        assert!(error.contains("verified #876 language bundle"));
    }

    #[test]
    fn compiler_binary_replacement_at_same_path_and_version_rebuilds_cache() {
        let compiler = temp_path("compiler");
        let vmfb = temp_path("compiler-digest").with_extension("vmfb");
        let weights = vec![AuxiliaryWeight {
            name: "weight".to_string(),
            storage: AuxiliaryWeightStorage::Bytes(1.0f32.to_ne_bytes().to_vec()),
            dtype: AuxiliaryWeightDType::Float32,
            shape: vec![1],
        }];
        std::fs::write(&compiler, b"compiler-build-a").unwrap();
        let first_generation =
            generation_identity_from_version(&compiler, &["--target=cuda"], "mlir", "v1").unwrap();
        assert!(first_generation.contains(&sha256_hex(b"compiler-build-a")));
        let first_contract =
            AuxiliaryArtifactContract::new("audio.main", "config=v1", &first_generation).unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(&vmfb, &first_contract, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-a")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();

        std::fs::write(&compiler, b"compiler-build-b").unwrap();
        let second_generation =
            generation_identity_from_version(&compiler, &["--target=cuda"], "mlir", "v1").unwrap();
        assert_ne!(first_generation, second_generation);
        let second_contract =
            AuxiliaryArtifactContract::new("audio.main", "config=v1", second_generation).unwrap();
        ensure_qualified_auxiliary_artifact(&vmfb, &second_contract, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-b")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 2);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-b");

        std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(compiler).ok();
    }

    #[test]
    fn decoded_audio_weights_reject_non_finite_values_before_native_upload() {
        let bf16_nan = bf16_to_f32(&0x7fc0u16.to_le_bytes());
        let f16_nan = f16_to_f32(&0x7e00u16.to_le_bytes());
        let f32_nan = f32_le_to_f32(&f32::NAN.to_le_bytes());
        let q4_infinite = dequantize_affine_bf16_sequential(
            &[0; 4],
            &0x7f80u16.to_le_bytes(),
            &[0; 2],
            1,
            1,
            4,
            8,
        )
        .unwrap();

        for (dtype, values) in [
            ("BF16", bf16_nan),
            ("F16", f16_nan),
            ("F32", f32_nan),
            ("Q4", q4_infinite),
        ] {
            let error = validate_finite_values(dtype, &values).unwrap_err();
            assert!(error.contains(dtype));
            assert!(error.contains("non-finite"));
            assert!(error.contains("flat index 0"));
        }
    }
}
