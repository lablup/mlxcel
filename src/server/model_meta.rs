// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Checkpoint facts for the b10621 `/v1/models` `meta` block (issue #1438).
//!
//! b10621 reads its model facts from the loaded GGUF; mlxcel derives the same
//! facts from the checkpoint directory: `config.json` for the geometry and
//! quantization, the safetensors headers for the parameter count and byte
//! size. Parsing every tensor header costs a few milliseconds once, so the
//! result is cached per canonical path for the life of the process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Facts reported under `meta` in the b10621 model object.
#[derive(Debug, Clone, Default)]
pub struct ModelFacts {
    /// `llama_vocab_type` analogue: 0 none, 1 SPM, 2 BPE (see
    /// [`vocab_type_code`]).
    pub vocab_type: i64,
    pub n_vocab: i64,
    /// Trained context length (`max_position_embeddings`).
    pub n_ctx_train: i64,
    /// Hidden size (`hidden_size`).
    pub n_embd: i64,
    /// Estimated parameter count (see module docs; quantized tensors are
    /// unpacked by their declared bit width).
    pub n_params: i64,
    /// Total bytes of the checkpoint's safetensors files.
    pub size: i64,
    /// Storage type string, e.g. `MLX Q4` / `MLX F16`.
    pub ftype: String,
}

/// The b10621 `llama_vocab_type` code for a tokenizer kind.
///
/// Upstream: 0 = none, 1 = SPM, 2 = BPE, 3 = WPM, 4 = UGM, 5 = RWKV. mlxcel
/// serves HuggingFace `tokenizer.json` (byte-level BPE), SentencePiece, and
/// tiktoken (BPE) tokenizers.
pub fn vocab_type_code(tokenizer: &crate::tokenizer::MlxcelTokenizer) -> i64 {
    use crate::tokenizer::MlxcelTokenizer;
    match tokenizer {
        MlxcelTokenizer::HuggingFace(_) => 2,
        MlxcelTokenizer::SentencePiece(_) => 1,
        MlxcelTokenizer::Tiktoken(_) => 2,
    }
}

fn facts_cache() -> &'static Mutex<HashMap<PathBuf, ModelFacts>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<PathBuf, ModelFacts>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Facts for the checkpoint at `model_path`, cached per path.
pub fn model_facts(model_path: &Path) -> ModelFacts {
    let key = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf());
    if let Ok(cache) = facts_cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return hit.clone();
    }
    let facts = derive_facts(model_path);
    if let Ok(mut cache) = facts_cache().lock() {
        cache.insert(key, facts.clone());
    }
    facts
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn derive_facts(model_path: &Path) -> ModelFacts {
    let config = read_json(&model_path.join("config.json"));
    let get_i64 = |key: &str| -> i64 {
        config
            .as_ref()
            .and_then(|c| c.get(key))
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
    };
    let quant_bits = config
        .as_ref()
        .and_then(|c| c.get("quantization"))
        .and_then(|q| q.get("bits"))
        .and_then(|b| b.as_u64());
    let ftype = match quant_bits {
        Some(bits) => format!("MLX Q{bits}"),
        None => "MLX F16".to_string(),
    };
    let (n_params, size) = scan_safetensors(model_path, quant_bits);
    ModelFacts {
        vocab_type: 0, // filled by the caller, which owns the tokenizer
        n_vocab: get_i64("vocab_size"),
        n_ctx_train: get_i64("max_position_embeddings"),
        n_embd: get_i64("hidden_size"),
        n_params,
        size,
        ftype,
    }
}

/// Sum the parameter count and byte size over every safetensors file.
///
/// A safetensors file starts with an 8-byte little-endian header length
/// followed by a JSON header carrying each tensor's dtype and shape; reading
/// it never touches the tensor data. A `uint32` tensor in a quantized
/// checkpoint packs `32 / bits` weights per element, which is how the
/// estimate recovers the true weight count; scale/bias tensors count at their
/// stored element count, a deliberate slight overcount that keeps the number
/// honest as "parameters stored", matching what the bytes on disk hold.
fn scan_safetensors(model_path: &Path, quant_bits: Option<u64>) -> (i64, i64) {
    let mut n_params: i64 = 0;
    let mut size: i64 = 0;
    let Ok(entries) = std::fs::read_dir(model_path) else {
        return (0, 0);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "safetensors") {
            continue;
        }
        if let Ok(meta) = path.metadata() {
            size += meta.len() as i64;
        }
        n_params += safetensors_params(&path, quant_bits).unwrap_or(0);
    }
    (n_params, size)
}

fn safetensors_params(path: &Path, quant_bits: Option<u64>) -> Option<i64> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes);
    // A corrupt header length must not allocate unbounded memory.
    if header_len > 256 * 1024 * 1024 {
        return None;
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let per_packed = quant_bits
        .map(|bits| (32 / bits.max(1)).max(1))
        .unwrap_or(1);
    let mut total: i64 = 0;
    for (name, tensor) in header.as_object()? {
        if name == "__metadata__" {
            continue;
        }
        let elems: i64 = tensor
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|dims| {
                dims.iter()
                    .filter_map(|d| d.as_i64())
                    .product::<i64>()
                    .max(0)
            })
            .unwrap_or(0);
        let dtype = tensor.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
        // Quantized weight payloads are stored packed in U32.
        if dtype == "U32" && per_packed > 1 {
            total += elems * per_packed as i64;
        } else {
            total += elems;
        }
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_model_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mlxcel-model-meta-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create dir");
        dir
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[i64])]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        let mut offset = 0i64;
        for (name, dtype, shape) in tensors {
            let elems: i64 = shape.iter().product();
            let width = match *dtype {
                "U32" | "F32" => 4,
                "F16" | "BF16" => 2,
                _ => 1,
            };
            let nbytes = elems * width;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + nbytes],
                }),
            );
            offset += nbytes;
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = std::fs::File::create(path).expect("create safetensors");
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&vec![0u8; offset as usize]).unwrap();
    }

    #[test]
    fn f16_checkpoint_counts_elements_directly() {
        let dir = temp_model_dir("f16");
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size": 1000, "max_position_embeddings": 2048, "hidden_size": 64}"#,
        )
        .unwrap();
        write_safetensors(
            &dir.join("model.safetensors"),
            &[("a.weight", "F16", &[1000, 64]), ("b.weight", "F16", &[64])],
        );
        let facts = derive_facts(&dir);
        assert_eq!(facts.n_params, 1000 * 64 + 64);
        assert_eq!(facts.n_vocab, 1000);
        assert_eq!(facts.n_ctx_train, 2048);
        assert_eq!(facts.n_embd, 64);
        assert_eq!(facts.ftype, "MLX F16");
        assert!(facts.size > 0);
    }

    #[test]
    fn quantized_checkpoint_unpacks_u32_by_bit_width() {
        let dir = temp_model_dir("q4");
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size": 8, "quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        // 16 packed u32 elements at 4 bits = 128 weights, plus 4 f16 scales.
        write_safetensors(
            &dir.join("model.safetensors"),
            &[("w.weight", "U32", &[4, 4]), ("w.scales", "F16", &[4])],
        );
        let facts = derive_facts(&dir);
        assert_eq!(facts.n_params, 16 * 8 + 4);
        assert_eq!(facts.ftype, "MLX Q4");
    }

    #[test]
    fn missing_checkpoint_yields_zeroed_facts() {
        let facts = derive_facts(Path::new("/definitely/not/a/model/dir"));
        assert_eq!(facts.n_params, 0);
        assert_eq!(facts.size, 0);
        assert_eq!(facts.ftype, "MLX F16");
    }
}
