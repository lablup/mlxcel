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

use std::fs;
use std::path::Path;

const EXPECTED_NATIVE: &[&str] = &[
    "ApertusModel@src/models/apertus.rs",
    "Cohere2Model@src/models/cohere2.rs",
    "CohereModel@src/models/cohere.rs",
    "DbrxModel@src/models/dbrx.rs",
    "DeepSeekModel@src/models/deepseek.rs",
    "DeepSeekV2Model@src/models/deepseek_v2.rs",
    "DeepSeekV32Model@src/models/deepseek_v32.rs",
    "DeepSeekV3Model@src/models/deepseek_v3.rs",
    "Ernie45Model@src/models/ernie4_5.rs",
    "ExaOneModel@src/models/exaone.rs",
    "Gemma2Model@src/models/gemma2.rs",
    "Gemma3Wrapper@src/models/gemma3.rs",
    "GemmaModel@src/models/gemma.rs",
    "Glm4Model@src/models/glm4.rs",
    "Glm4MoeLiteModel@src/models/glm4_moe_lite.rs",
    "GraniteModel@src/models/granite.rs",
    "HeliumModel@src/models/helium.rs",
    "HunyuanV1DenseModel@src/models/hunyuan_v1_dense.rs",
    "InternLM2Model@src/models/internlm2.rs",
    "InternLM3Model@src/models/internlm3.rs",
    "Llama3Model@src/models/llama3.rs",
    "Llama4Wrapper@src/models/llama4.rs",
    "MiMoModel@src/models/mimo.rs",
    "MiniCPM3Model@src/models/minicpm3.rs",
    "Mistral4Model@src/models/mistral4.rs",
    "MixtralModel@src/models/mixtral.rs",
    "OLMo2Model@src/models/olmo2.rs",
    "OLMo3Model@src/models/olmo3.rs",
    "OlmoeModel@src/models/olmoe.rs",
    "Phi3Model@src/models/phi3.rs",
    "PhiMoeModel@src/models/phimoe.rs",
    "PhixtralModel@src/models/phixtral.rs",
    "Qwen25VLModel@src/vision/qwen2_5_vl.rs",
    "Qwen2MoeModel@src/models/qwen2_moe.rs",
    "Qwen2VLModel@src/models/qwen2_vl.rs",
    "Qwen2VLModel@src/vision/qwen2_vl.rs",
    "Qwen3Model@src/models/qwen3.rs",
    "Qwen3MoeModel@src/models/qwen3_moe.rs",
    "Qwen3OmniMoeModel@src/vision/qwen3_omni_moe.rs",
    "Qwen3VLModel@src/models/qwen3_vl.rs",
    "Qwen3VLModel@src/vision/qwen3_vl.rs",
    "Qwen3VLMoeModel@src/models/qwen3_vl_moe.rs",
    "Qwen3VLMoeModel@src/vision/qwen3_vl_moe.rs",
    "SeedOssModel@src/models/seed_oss.rs",
    "SmolLM3Model@src/models/smollm3.rs",
    "StarCoder2Model@src/models/starcoder2.rs",
    "TeleChat3Model@src/models/telechat3.rs",
    "TensorParallelErnie45Model@src/distributed/tensor_parallel/llama_runtime.rs",
    "TensorParallelGemma3Model@src/distributed/tensor_parallel/llama_runtime.rs",
    "TensorParallelHunyuanV1DenseModel@src/distributed/tensor_parallel/llama_runtime.rs",
    "TensorParallelLlamaModel@src/distributed/tensor_parallel/llama_runtime.rs",
    "TensorParallelQwen35Model@src/distributed/tensor_parallel/llama_runtime.rs",
    "TensorParallelQwen3Model@src/distributed/tensor_parallel/llama_runtime.rs",
    "YoutuLanguageModel@src/models/youtu_vl_lm.rs",
];

const EXPECTED_DELEGATING: &[&str] = &[
    "DeepSeekOcr2VlModel@src/vision/deepseekocr_2.rs",
    "DeepSeekOcrVlModel@src/vision/deepseekocr.rs",
    "DeepSeekVl2VlModel@src/vision/deepseek_vl2.rs",
    "GlmMoeDsaModel@src/models/glm_moe_dsa.rs",
    "GotOcrVlModel@src/vision/got_ocr.rs",
    "GraniteVisionVLModel@src/vision/granite_vision.rs",
    "Idefics2Model@src/vision/idefics2.rs",
    "InternVLChatVLM@src/vision/internvl.rs",
    "JinaVlmModel@src/vision/jina_vlm.rs",
    "KimiVLModel@src/vision/kimi_vl.rs",
    "LlmJpTextModel@src/vision/llmjp_vl_text.rs",
    "LlmJpVlModel@src/vision/llmjp_vl.rs",
    "LocateAnythingVLM@src/vision/locateanything.rs",
    "MageVlModel@src/vision/mage_vl.rs",
    "MiniCPMV46VLModel@src/vision/minicpmv4_6_vl.rs",
    "MuseGlimmerVlmModel@src/vision/muse_glimmer_vlm.rs",
    "Phi3VLModel@src/vision/phi3_vl.rs",
    "Phi4SigLipVLModel@src/vision/phi4_siglip_vl.rs",
    "Qwen35VLModel@src/vision/qwen3_5_vl.rs",
    "Qwen4ExpVLModel@src/vision/qwen4_exp_vl.rs",
    "SmolVLMModel@src/vision/smolvlm.rs",
    "VisionLanguageModel@src/vision/mod.rs",
    "YoutuVLModel@src/vision/youtu_vl.rs",
];

const EXPECTED_PER_ROW_FALLBACK: &[&str] = &[
    "BaichuanModel@src/models/baichuan.rs",
    "BailingMoeModel@src/models/bailing_moe.rs",
    "BitNetModel@src/models/bitnet.rs",
    "Cohere2MoeModel@src/models/cohere2_moe.rs",
    "CohereCompassModel@src/vision/cohere_compass.rs",
    "CohereCompassTextModel@src/models/cohere_compass.rs",
    "Dots1Model@src/models/dots1.rs",
    "DotsOcrVlModel@src/vision/dots_ocr.rs",
    "Ernie45MoeModel@src/models/ernie4_5_moe.rs",
    "Ernie45MoeVlModel@src/vision/ernie4_5_moe_vl.rs",
    "Ernie45MoeVlTextModel@src/models/ernie4_5_moe_vl.rs",
    "ExaoneMoeModel@src/models/exaone_moe.rs",
    "Gemma3nModel@src/models/gemma3n.rs",
    "Gemma3nVLModel@src/vision/gemma3n_vl.rs",
    "Gemma4VLModel@src/vision/gemma4_vl.rs",
    "Gemma4Wrapper@src/models/gemma4.rs",
    "Glm4MoeModel@src/models/glm4_moe.rs",
    "Glm4vModel@src/vision/glm4v.rs",
    "Glm4vMoeModel@src/vision/glm4v_moe.rs",
    "Glm4vMoeTextModel@src/models/glm4v_moe.rs",
    "Glm4vTextModel@src/models/glm4v.rs",
    "GlmOcrModel@src/vision/glm_ocr.rs",
    "Gpt2Model@src/models/gpt2.rs",
    "GptBigCodeModel@src/models/gpt_bigcode.rs",
    "GptNeoxModel@src/models/gpt_neox.rs",
    "HunyuanMoeModel@src/models/hunyuan_moe.rs",
    "HunyuanVlModel@src/vision/hunyuan_vl.rs",
    "HunyuanVlTextModel@src/models/hunyuan_vl.rs",
    "InklingVlModel@src/vision/inkling_vl.rs",
    "JinaVlmTextModel@src/models/jina_vlm.rs",
    "KimiK3VLModel@src/vision/kimi_k3_vl.rs",
    "KlearModel@src/models/klear.rs",
    "Lfm2VlModel@src/vision/lfm2_vl.rs",
    "Llama4CxxModel@src/models/llama4.rs",
    "MiniCPMModel@src/models/minicpm.rs",
    "MiniCPMOVLModel@src/vision/minicpmo_vl.rs",
    "MiniMaxM3Model@src/models/minimax_m3.rs",
    "MiniMaxM3VlModel@src/vision/minimax_m3_vl.rs",
    "MiniMaxModel@src/models/minimax.rs",
    "MllamaVLModel@src/vision/mllama_vl.rs",
    "Molmo2Model@src/models/molmo2.rs",
    "Molmo2VLModel@src/vision/molmo2_vl.rs",
    "MolmoModel@src/models/molmo.rs",
    "MolmoPointVLModel@src/vision/molmo_point_vl.rs",
    "MolmoVLModel@src/vision/molmo_vl.rs",
    "Moondream2Model@src/models/moondream2.rs",
    "Moondream2VLModel@src/vision/moondream2_vl.rs",
    "Moondream3Model@src/models/moondream3.rs",
    "Moondream3VLModel@src/vision/moondream3_vl.rs",
    "MuseGlimmerTextWrapper@src/models/muse_glimmer.rs",
    "NanoChatModel@src/models/nanochat.rs",
    "NemotronHNanoOmniVlModel@src/vision/nemotron_h_nano_omni_vl.rs",
    "NemotronModel@src/models/nemotron.rs",
    "OlmoModel@src/models/olmo.rs",
    "OpenElmModel@src/models/openelm.rs",
    "PaddleOcrTextModel@src/models/paddleocr_vl.rs",
    "PaddleOcrVlModel@src/vision/paddleocr_vl.rs",
    "Phi3SmallModel@src/models/phi3small.rs",
    "PhiModel@src/models/phi.rs",
    "PipelineServerModel@src/distributed/pipeline/server_runtime.rs",
    "Qwen35Model@src/models/qwen3_5.rs",
    "Qwen4ExpModel@src/models/qwen4_exp.rs",
    "SolarOpenModel@src/models/solar_open.rs",
    "StableLMModel@src/models/stablelm.rs",
    "TensorParallelGemma4Model@src/distributed/tensor_parallel/llama_runtime.rs",
];

#[test]
fn every_batching_language_model_has_a_pinned_decode_classification() {
    let mut native = Vec::new();
    let mut delegating = Vec::new();
    let mut fallback = Vec::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in ["src/models", "src/vision", "src/distributed"] {
        visit_rs_files(&root.join(relative), &mut |path| {
            classify_file(root, path, &mut native, &mut delegating, &mut fallback)
        });
    }
    native.sort();
    delegating.sort();
    fallback.sort();
    assert_eq!(native, EXPECTED_NATIVE, "native decode list drifted");
    assert_eq!(
        delegating, EXPECTED_DELEGATING,
        "delegating decode list drifted"
    );
    assert_eq!(
        fallback, EXPECTED_PER_ROW_FALLBACK,
        "per-row fallback list drifted"
    );
}

#[test]
fn issue_2556_dense_families_use_the_shared_native_route_and_attention() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "src/models/apertus.rs",
        "src/models/cohere.rs",
        "src/models/cohere2.rs",
        "src/models/ernie4_5.rs",
        "src/models/exaone.rs",
        "src/models/gemma.rs",
        "src/models/gemma2.rs",
        "src/models/glm4.rs",
        "src/models/granite.rs",
        "src/models/hunyuan_v1_dense.rs",
        "src/models/internlm2.rs",
        "src/models/internlm3.rs",
        "src/models/mimo.rs",
        "src/models/olmo2.rs",
        "src/models/olmo3.rs",
        "src/models/phi3.rs",
        "src/models/seed_oss.rs",
        "src/models/smollm3.rs",
        "src/models/starcoder2.rs",
        "src/models/telechat3.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("model source is readable");
        assert!(
            source.contains("route_batched_decode(self"),
            "{relative} bypasses the B=0/B=1/native routing contract"
        );
        assert!(
            source.contains("forward_split_attention("),
            "{relative} bypasses shared split attention"
        );
        if relative != "src/models/qwen2_vl.rs" {
            assert!(
                source.contains("shape[1] != 1 || self.has_deepstack_state()"),
                "{relative} must retain per-row decode for model-global DeepStack state (#2567)"
            );
        }
        assert!(
            source.contains("collect_layer_caches("),
            "{relative} bypasses contextual row/layer cache validation"
        );
        assert!(
            source.contains("uniform_layer_metadata("),
            "{relative} bypasses contextual family/layer metadata validation"
        );
    }
}

#[test]
fn issue_2556_vlm_wrappers_delegate_only_to_their_text_tower() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "src/vision/granite_vision.rs",
        "src/vision/phi3_vl.rs",
        "src/vision/phi4_siglip_vl.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("wrapper source is readable");
        let start = source
            .find("fn forward_batched(")
            .expect("wrapper overrides forward_batched");
        let open = source[start..]
            .find('{')
            .map(|offset| start + offset)
            .expect("wrapper forward_batched body");
        let close = matching_brace(&source, open);
        let body = &source[open..=close];
        assert!(
            body.contains("LanguageModel::forward_batched(&self.text_model"),
            "{relative} does not delegate text-only decode directly"
        );
        assert!(
            !body.contains("vision_"),
            "{relative} runs vision work in decode"
        );
    }
}

#[test]
fn issue_2562_qwen_vl_towers_and_wrappers_use_native_sequence_aware_decode() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "src/models/qwen2_vl.rs",
        "src/models/qwen3_vl.rs",
        "src/models/qwen3_vl_moe.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("tower source is readable");
        assert!(
            source.contains("route_batched_decode_with_ids(self"),
            "{relative} bypasses the sequence-aware B=0/B=1 route"
        );
        assert!(
            source.contains("batched_decode_positions(seq_ids, &offsets)"),
            "{relative} bypasses per-sequence MRoPE positions"
        );
        assert!(
            source.contains("forward_split_attention("),
            "{relative} bypasses shared split attention"
        );
    }

    for relative in [
        "src/vision/qwen2_vl.rs",
        "src/vision/qwen2_5_vl.rs",
        "src/vision/qwen3_vl.rs",
        "src/vision/qwen3_vl_moe.rs",
        "src/vision/qwen3_omni_moe.rs",
    ] {
        let source = fs::read_to_string(root.join(relative)).expect("wrapper source is readable");
        assert!(
            source.contains("forward_batched_decode_with_ids(input_ids"),
            "{relative} does not delegate identified decode to its native text tower"
        );
    }
}

fn visit_rs_files(dir: &Path, visit: &mut impl FnMut(&Path)) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            visit_rs_files(&path, visit);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs")
            && path.file_name().and_then(|name| name.to_str()) != Some("tests.rs")
            && !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_tests.rs"))
        {
            visit(&path);
        }
    }
}

fn classify_file(
    root: &Path,
    path: &Path,
    native: &mut Vec<String>,
    delegating: &mut Vec<String>,
    fallback: &mut Vec<String>,
) {
    let source =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let relative = relative_source_path(root, path);
    let needles = [
        "impl LanguageModel for ",
        "impl mlxcel_core::generate::LanguageModel for ",
    ];
    let mut cursor = 0;
    while let Some((start, needle)) = needles
        .iter()
        .filter_map(|needle| {
            source[cursor..]
                .find(needle)
                .map(|offset| (cursor + offset, *needle))
        })
        .min_by_key(|(offset, _)| *offset)
    {
        let type_start = start + needle.len();
        let open = source[type_start..]
            .find('{')
            .map(|offset| type_start + offset)
            .expect("impl body");
        let type_name = source[type_start..open]
            .split_whitespace()
            .next()
            .expect("impl type");
        let close = matching_brace(&source, open);
        let block = &source[open + 1..close];
        cursor = close + 1;

        let key = format!("{type_name}@{relative}");
        match classify_impl(&key, block).unwrap_or_else(|message| panic!("{message}")) {
            Some(DecodeClassification::Native) => native.push(key),
            Some(DecodeClassification::Delegating) => delegating.push(key),
            Some(DecodeClassification::Fallback) => fallback.push(key),
            None => {}
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DecodeClassification {
    Native,
    Delegating,
    Fallback,
}

fn classify_impl(key: &str, block: &str) -> Result<Option<DecodeClassification>, String> {
    if method_body(block, "supports_batching").is_some_and(literal_false) {
        return Ok(None);
    }
    let has_batched_override =
        block.contains("fn forward_batched(") || block.contains("fn forward_batched_with_ids(");
    match method_body(block, "batched_decode_kind") {
        Some(body) if body.contains("BatchedDecodeKind::Native") => {
            if has_batched_override {
                Ok(Some(DecodeClassification::Native))
            } else {
                Err(format!(
                    "{key} declares native batched decode without a batched forward override"
                ))
            }
        }
        Some(_) => {
            if has_batched_override {
                Ok(Some(DecodeClassification::Delegating))
            } else {
                Err(format!(
                    "{key} declares delegating batched decode without a batched forward override"
                ))
            }
        }
        None => Ok(Some(DecodeClassification::Fallback)),
    }
}

fn method_body<'a>(block: &'a str, name: &str) -> Option<&'a str> {
    let start = block.find(&format!("fn {name}"))?;
    let open = block[start..].find('{').map(|offset| start + offset)?;
    let close = matching_brace(block, open);
    Some(block[open + 1..close].trim())
}

fn literal_false(body: &str) -> bool {
    body.lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<String>()
        .trim()
        == "false"
}

fn matching_brace(source: &str, open: usize) -> usize {
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return open + offset;
                }
            }
            _ => {}
        }
    }
    panic!("unclosed impl block at byte {open}")
}

fn relative_source_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("source under repository root")
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[test]
fn native_marker_without_a_batched_override_is_rejected() {
    let block = r#"
        fn supports_batching(&self) -> bool { true }

        fn batched_decode_kind(&self) -> BatchedDecodeKind {
            BatchedDecodeKind::Native
        }
    "#;
    assert_eq!(
        classify_impl("DroppedOverride@synthetic.rs", block),
        Err(
            "DroppedOverride@synthetic.rs declares native batched decode without a batched forward override"
                .to_string()
        )
    );
}

#[test]
fn delegating_marker_without_a_batched_override_is_rejected() {
    let block = r#"
        fn supports_batching(&self) -> bool { true }

        fn batched_decode_kind(&self) -> BatchedDecodeKind {
            self.text_model.batched_decode_kind()
        }
    "#;
    assert_eq!(
        classify_impl("DroppedWrapperOverride@synthetic.rs", block),
        Err(
            "DroppedWrapperOverride@synthetic.rs declares delegating batched decode without a batched forward override"
                .to_string()
        )
    );
}

#[test]
fn pinned_source_paths_use_portable_separators() {
    let root = Path::new("workspace");
    let path = root.join("src").join("models").join("qwen.rs");
    assert_eq!(relative_source_path(root, &path), "src/models/qwen.rs");
}

#[test]
fn deepseek_ocr_wrappers_delegate_text_only_batched_decode() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative in ["src/vision/deepseekocr.rs", "src/vision/deepseekocr_2.rs"] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(
            source.contains("self.text_model.batched_decode_kind()"),
            "{relative} must expose the DeepSeek text model's decode class"
        );
        assert!(
            source.contains(
                "self.text_model\n            .forward_batched(input_ids, batch_caches, mask)"
            ),
            "{relative} must delegate text-only batches without running the vision towers"
        );
    }
}
