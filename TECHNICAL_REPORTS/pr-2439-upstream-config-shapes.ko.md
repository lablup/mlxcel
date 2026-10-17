# PR #2439: 업스트림 config 형태, alias, key 레이아웃 수용

**작성일**: 2026-10-17
**상태**: CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. 후속 작업 두 건은 미수행 (「미해결 사항」 참조)
**위험도**: 중간 (Qwen3, GraniteMoeHybrid, Nemotron-H, Muse Glimmer, LFM2-VL, Moondream3, EOS id 읽기의 config 파싱을 변경함. Qwen3와 GraniteMoeHybrid는 이전에 오류 없이 잘못된 theta로 디코딩했음)

## 요약

#2384를 닫는다. 업스트림 mlx-lm과 mlx-vlm은 mlxcel이 로드 시 거부하거나 잘못된 값으로 받아들이던 config 형태, `model_type` 표기, 체크포인트 key 레이아웃을 수용한다. 이 PR은 이슈의 여덟 항목을 한 PR에서 모두 고친다: 중첩된 `rope_parameters`, transformers가 저장한 Nemotron-H config, 누락된 `model_type` alias, DeepSeek-V4 양자화 embedding plane, flat 구조의 text-only Muse Glimmer config, 평탄화된 OptiQ key 레이아웃, Moondream3 mlx-vlm 양자화 변환본, EOS id 출처의 합집합.

## 문제 정의

이슈는 여덟 가지 공백을 나열한다. 일곱 건은 로드 시 명확히 실패하고, 항목 1만 조용히 틀린다.

- **항목 1, 중첩 `rope_parameters`:** transformers 5는 `rope_theta`와 `rope_scaling`을 `rope_parameters` 아래에 중첩하고 flat key를 없앤다. 그러면 `qwen3::ModelArgs`와 `granitemoehybrid::ModelArgs`는 serde 기본 theta인 10000으로 대체된다. 모델은 짧은 길이에서는 자연스럽게 디코딩하고 긴 길이에서 어긋난다.
- **항목 2, Nemotron-H:** `num_hidden_layers`가 필수였고, `layers_block_type`만 읽고 `layer_types` alias는 읽지 않았으며, 블록 이름 맵이 `full_attention`, `linear_attention`, `conv`를 거부했고, `time_step_limit` visitor는 transformers의 `{"__float__": "..."}` 태그에서 실패했다. 숫자가 아닌 두 번째 원소는 `+inf`가 되어 `-Infinity`와 `NaN`도 `+inf`가 되었다.
- **항목 3, alias:** `kimi_k2`, `joyai_llm_flash`, `minimax_m2`, `xverse`, `qwen3_5_text`, `qwen3_5_moe_text`가 `detect_model_type`에 없어 "Unsupported model type"으로 실패했다.
- **항목 4, DeepSeek-V4:** legacy namespace 변환이 `embed.weight`와 `head.weight`는 바꿨지만 `.scales`와 `.biases`는 바꾸지 않아, legacy namespace MLX 양자화본이 필수 key 검사에서 실패했다.
- **항목 5, Muse Glimmer:** `MuseGlimmerConfig`가 `text_config`를 필수로 요구했고, detection은 `muse_glimmer`를 항상 VLM으로 보냈다. mlx-lm text-only 변환본(flat config, `model.*`와 `lm_head.*` key)은 로드할 수 없었다. `MuseRopeParameters`는 `rope_theta`만 파싱해서 `linear`, `yarn`, `llama3` `rope_type`이 무시되었다.
- **항목 6, OptiQ 레이아웃:** `lfm2_text_weights`가 `language_model.*` key만 유지해서 평탄화된 LFM2-VL은 text tower를 잃었다. `normalize_muse_weight_key`는 bare `model.layers.*`를 `model.language_model` 아래로 매핑하지 않고 그대로 통과시켰다.
- **항목 7, Moondream3:** `rewrite_moondream3_weight_key`가 원래 레이아웃(`text.blocks.*`, `text.wte`, `vision.blocks.*`)만 처리하고 mlx-vlm 양자화 레이아웃(`text.model.blocks.*`, `vision.encoder.*`)은 처리하지 못했다. vision tower는 group 64, 4-bit를 하드코딩했다.
- **항목 8, EOS id:** `read_eos_token_ids`가 `generation_config.json`, `tokenizer_config.json`, `config.json` 중 처음으로 비어 있지 않은 값을 반환해서, generation config에 stop id가 하나만 있는 체크포인트는 다른 id에서 멈추지 않았다.

## 기술적 선택과 그 이유

### Deserialize 안에서 rope_parameters 끌어올리기

`qwen3::ModelArgs`와 `granitemoehybrid::ModelArgs`는 이제 `#[serde(remote = "Self")]`로 derive하고, `rope_utils::deserialize_lifting_rope_parameters`를 거치는 `Deserialize`를 직접 구현한다. 모든 `from_value`와 `from_str` 지점이 끌어올리기를 적용받는다. 이슈는 private wire struct를 제안했으나, `remote = "Self"`는 필드 목록을 복사하지 않고도 같은 효과를 낸다. 이슈는 지점마다 lifter를 호출하는 방식을 기각했는데, 다음에 추가되는 파싱 지점이 놓치기 때문이다.

lifter는 Mage-VL config에서 `rope_utils`로 옮겼고 Mage-VL은 이를 re-export한다. flat `rope_theta`가 중첩 값보다 우선한다.

커버리지 점검: mlx-lm 0.32.2가 `rope_parameters`를 끌어올리는 계열(exaone_moe, gemma4_text, glm_moe_dsa, laguna, lfm2/lfm2_moe, mellum, ministral3, mistral4, muse_glimmer, qwen3_5)은 mlxcel에서 이미 중첩 블록을 읽는다. 공백은 Qwen3와 GraniteMoeHybrid뿐이었다.

### Text-only Muse Glimmer

flat config는 자기 자신을 `text_config`로 삼는다. vision이 없는 config(vision config 또는 vision weight가 없음)는 새 text-only `ModelType::MuseGlimmer`로 로드된다. `rope_parameters` 방식은 `RopeScalingKind`로 해석한다. text-only 모델은 계열 가드(adapter 불가, fp16 KV만)를 공유한다. 리뷰 후에는 DFlash round loop에 text-only wrapper용 target arm이 없으므로, startup이 이 모델에 대한 모든 drafter도 거부한다.

### OptiQ vision sidecar

두 OptiQ 체크포인트는 vision tower를 `optiq/optiq_vision.safetensors`에 두고, `config.json`의 `optiq_vision.sidecar`가 이를 가리킨다. shard 로더는 하위 디렉터리 shard를 거부하므로, VLM weight 로딩이 sidecar를 직접 병합한다(상대 `.safetensors` 경로만). 코어 shard 검증기는 바꾸지 않았다. 리뷰 후에는 sidecar가 없을 때 로드 오류 대신 경고를 낸다. text-only 로드는 sidecar를 읽지 않고, VLM 로드는 누락된 첫 vision tensor에서 여전히 실패한다.

## 변경 사항

- **Nemotron-H** (`src/models/nemotron_h.rs`): `num_hidden_layers`는 선택 항목이고 `layer_types`는 `layers_block_type`의 alias다. transformers 5.13 이름은 pattern code로 매핑되며, 알 수 없는 이름은 허용 목록을 보여 준다. `time_step_limit`은 `__float__` 태그를 디코딩하고 다른 비숫자는 거부한다. 로더의 `sanitize_config_json`이 태그의 `"Infinity"`를 `"1e38"`로 바꾸므로 숫자 형태의 태그 문자열도 받는다.
- **Alias** (`src/models/detection.rs`, `registry.rs`): `kimi_k2`, `joyai_llm_flash`, `minimax_m2`, `xverse`(`llama` arm 공유), `qwen3_5_text`, `qwen3_5_moe_text`를 detection과 registry key에 추가했다. DeepSeek-V3 alias 두 개는 `MLA_LATENT_CACHE_FAMILIES`에도 추가해서, 양자화 KV 모드가 이들에서도 fp16으로 결정된다.
- **DeepSeek-V4** (`src/models/deepseek_v4_sanitize.rs`): embed와 head를 stem 기준으로 이름을 바꾼다.
- **LFM2-VL과 Muse Glimmer key 맵**: 평탄화된 OptiQ text root를 `model.language_model` 아래로 매핑(Muse)하고 text key로 받아들인다(LFM2-VL).
- **Moondream3** (`src/loading/vlm_special.rs`, `src/vision/encoders/moondream3.rs`): 두 key 레이아웃을 모두 받고, 양자화 group size와 bits는 config에서 읽는다.
- **EOS id** (`src/loading/mod.rs`): `read_eos_token_ids`는 출처들의 중복 제거된 합집합을 반환한다.
- **문서**: `docs/supported-models.md`에 새 alias를 기록했다.

## 동작 변경

- `read_eos_token_ids`가 출처들의 합집합을 반환한다. 첫 출처가 우선한다고 가정한 기존 테스트(Kimi K3 tokenizer pin 포함)를 갱신했다.
- Qwen3와 GraniteMoeHybrid에서 이전에 serde 기본 theta 10000으로 로드되던 config는 이제 중첩 theta를 읽는다.
- `ModelType::MuseGlimmer`는 vision이 없는 Muse config를 위한 새 text-only 경로다.

## 검증

GB10, `--profile test-fast --features cuda`, `MLXCEL_SDPA_DETERMINISTIC=1`, greedy. 체크포인트는 `~/models_archive/mlx/<name>`에서 읽었다(실행 중 호스트가 `~/models/mlx`를 이곳으로 옮김). 기본 경로 Muse 실행은 `/mnt/models-10/mlx`를 통해 같은 고정 `muse-glimmer-30b-4bit`를 사용했다. transformers-5 사본은 원본 체크포인트에서 `config.json`만 바꾼 것이다. 긴 프롬프트는 CONTRIBUTING.md(22 KB) 뒤에 질문을 붙인 것이며 `generate -n 64 --temp 0`으로 실행했다.

| 체크포인트 | 결과 |
|------------|------|
| `qwen3-0.6b-4bit` (transformers-5 사본) | main은 사본을 깨진 출력으로 디코딩한다. 브랜치는 원본과 바이트 단위로 같게 디코딩하며, 원본은 main과 브랜치에서 동일하다. |
| `granite-4.0-3b-vision-4bit` (granitemoehybrid text, theta 1e7, RoPE) | 같은 결과. `granite-4.0-h-*` 체크포인트는 NoPE라 이 버그를 보여 줄 수 없다. |
| `nvidia-nemotron-3-nano-30b-a3b-4bit` (`layer_types`, `__float__`, `num_hidden_layers` 없음) | main은 "missing field num_hidden_layers"로 실패한다. 브랜치 출력은 원본과 동일하다. |
| `mlx-community/XVERSE-7B-Chat-4bit` | main은 "Unsupported model type: xverse"로 실패한다. 브랜치는 자연스러운 출력을 낸다 ("Seoul is the capital and most populous city of South Korea..."). |
| `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` | main은 `model.embed_tokens` 누락으로 실패한다. 브랜치는 "Paris"로 답하고 빨간 사각형, 파란 원, 초록 삼각형을 나열한다. |
| `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` | main은 알 수 없는 weight root로 실패한다. 브랜치는 텍스트 프롬프트("The capital of France is Paris.")와 이미지 프롬프트에 답한다. flat text-only 사본은 VLM 텍스트 경로와 동일한 토큰으로 디코딩한다. |
| `mlx-community/Muse-Glimmer-30B-4bit` (기본 경로) | 텍스트와 이미지 프롬프트 각각 greedy 64토큰이 main과 브랜치에서 동일하다. |
| `beshkenadze/moondream3-preview-mlx-4bit` | main은 `text.blocks.0.mlp.fc1.weight` 누락으로 실패한다. 브랜치는 shapes 이미지를 올바르게 캡션한다. |

새로 추가하거나 변경한 단위 테스트 30개가 모두 통과한다. 수정을 무력화하면 그중 26개가 실패한다. 네 개는 예상대로 그대로 통과한다: flat이 중첩보다 우선하는 pin 두 개, 기본 RoPE pin, 그리고 비숫자 거부 케이스(실제 `main`에서는 +inf로 파싱됨). Kimi-K2, JoyAI, MiniMax-M2는 이 호스트에 올라가지 않아 detection 테스트만 있다. clippy(lib+tests, `-D warnings`), fmt, 다섯 가지 contract check가 통과한다.

## 리뷰 중 변경

두 번째 커밋(`0c846ab4`)은 text-only Muse에 대한 drafter를 거부하고, OptiQ sidecar 누락을 경고로 낮추며, Muse startup guard fixture에 `vision_config`를 추가해 VLM 경로에 머물게 한다. `text_only_muse_glimmer_refuses_a_drafter`가 새 가드를 검증한다.

## 미해결 사항

- mlx-lm도 끌어올리지 않는 다른 flat-`rope_theta` 계열 (예: `qwen3_moe`).
- `layer_types`만 가진 Nemotron-H config의 `kv_arch` 메모리 추정.

## 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 40 |
| 추가 줄 | +1568 |
| 삭제 줄 | -140 |
| 새로 추가하거나 변경한 단위 테스트 | 30 |

| 해시 | 메시지 |
|------|--------|
| `6ece3f6f` | fix(loading): accept upstream config shapes, aliases and key layouts |
| `0c846ab4` | fix(server): refuse a drafter for text-only Muse Glimmer |
