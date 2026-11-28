# PR #2448: 모든 모델 계열에서 transformers 5 rope_parameters 끌어올리기

**작성일**: 2026-10-19
**상태**: CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. PR은 열려 있음. "미해결 사항"의 항목은 계속 열려 있음

## 요약

#2443을 닫는다. transformers 5는 `rope_theta`, `rope_scaling`, `partial_rotary_factor`를 중첩된 `rope_parameters` 블록 안에만 저장한다. 지금까지 Qwen3와 GraniteMoeHybrid만 이를 끌어올렸고(#2439), 평평한(flat) 키를 읽는 나머지 struct는 오류 없이 serde 기본 base로 되돌아갔다. 이 PR은 설정 타입에서 끌어올리기를 적용한다. struct 86개에 매크로 한 줄씩 추가하고, serde 밖에서 파싱하는 호출 지점도 함께 처리한다.

## 1. 문제 정의

- serde 기본값이 있는 flat `rope_theta`를 읽는 설정 struct(llama3, qwen3_moe, gemma2, phi3, olmo2는 10000)는 transformers 5 체크포인트를 오류 없이 로드하고, 짧은 프롬프트에서는 유창하게 디코딩하지만 긴 프롬프트에서는 어긋난다. Llama 3 설정은 `rope_type: "llama3"` 스케일링도 잃는다.
- `main`에서 Llama 3.2 1B 또는 Qwen3-30B-A3B의 transformers-5 사본은 20 KB 프롬프트에서 깨진 출력을 만든다. Llama 사본은 `and, and, ...`로, Qwen3 사본은 `. The. The. ...`로 퇴화한다.
- mlx-lm 0.32.2는 exaone_moe, gemma4, glm_moe_dsa, laguna, lfm2, mellum, ministral3, mistral4, muse_glimmer, qwen3_5만 끌어올리며 mlxcel은 이미 이들을 처리하므로, 다른 계열은 parity로는 수정을 얻지 못한다.
- EXAONE-MoE와 Mistral 4는 중첩된 base가 우선했는데, flat 키가 우선하는 다른 계열과 일관되지 않았다.
- `src/loading/vlm_qwen.rs`에 임시방편 호출 지점 끌어올리기가 두 곳 있었다(GLM-OCR, GLM-4V MoE).

## 2. 기술적 선택과 그 이유

- **타입이 병목 지점이다.** PR 작성자는 모든 지점이 이미 호출하는 공유 파싱 헬퍼를 찾았으나 없었다. 약 125개의 계열별 `load` 함수가 `config.json`을 직접 읽고, VLM은 블록을 `text_config` 아래에 중첩한다. 이슈는 `sanitize_config_json`의 최상위 끌어올리기를 두 이유로 배제한다(대부분의 계열 `load` 함수가 건너뛰는 문자열 치환이며, `text_config`를 놓친다).
- **`mlxcel-core`의 공유 lifter.** `lift_rope_parameters` / `deserialize_lifting_rope_parameters`를 `mlxcel_core::rope_parameters`로 옮기고(`models::rope_utils`에서 re-export), DFlash drafter 설정도 사용할 수 있게 한다. `impl_rope_lifting_deserialize!(Type)` 매크로가 미처리 struct마다 `#[serde(remote = "Self")]`와 한 줄을 제공한다. VLM 텍스트 설정과 Qwen3-Omni speech 설정 3개를 포함해 86개 struct를 다룬다. 파이프라인과 tensor-parallel 지점도 같은 타입을 파싱하므로 포함된다. 어떤 struct에도 필드가 추가되지 않는다.
- **Lifter 규칙.**
  - null이 아닌 flat 키가 우선한다.
  - `partial_rotary_factor`를 끌어올린다.
  - 레이어 종류별 형식, 즉 값이 모두 객체인 블록을 읽는다. Gemma 3 / Gemma 3n / OLMo 3는 `full_attention`이 base와 스케일링을, `sliding_attention`이 `rope_local_base_freq`를 준다. DeepSeek-V4는 `main`이 base를, `compress`가 `compress_rope_theta`와 YaRN `rope_scaling`을 준다.
  - 기본이 아닌 스킴(Llama 3 `llama3` 포함) 또는 `mrope_section` 같은 MRoPE 추가 값으로 `rope_scaling`을 만든다.
  - 스킴은 `type` 아래에만 넣는다(`type`을 먼저 읽고 `rope_type`을 읽는다). deepseek_v32 / deepseek_v4 / nemotron의 `alias = "type", alias = "rope_type"` 필드가 둘을 모두 적은 블록을 거부하기 때문이다.
- **serde 밖의 호출 지점도 공유 lifter를 쓴다.** `DFlashConfig::from_json`, Idefics2 Mistral 기본값(기존에는 중첩 base를 가렸다), 그리고 `vlm_qwen.rs`의 임시 끌어올리기 두 곳이다. `JinaVlmTextConfig::from_json`은 중첩 base로 폴백하고, Phi-4 텍스트 필드 복사는 이제 `rope_parameters`를 함께 옮긴다.
- **모든 곳에서 flat 우선.** 이슈의 지시대로 EXAONE-MoE와 Mistral 4도 flat 키가 우선한다. transformers 5 자체는 중첩 값에 우선순위를 주지만, 둘이 다른 값을 가진 설정에서만 차이가 나며 참조 체크포인트 중에는 그런 것이 없다.
- **`exaone4`.** `rope_type`만 확인하던 유일한 reader였으며 이제 `type`도 받아들인다. 기본 스킴으로 저장한 transformers 5 설정에는 `rope_scaling`이 없으므로 `#[serde(default)]`로 바꿨다.

## 3. 변경 내용

- `src/lib/mlxcel-core/src/rope_parameters.rs`(신규, 187줄), `rope_parameters_tests.rs`(신규, lifter 단위 테스트), `lib.rs`: 공유 lifter와 매크로.
- `src/models/rope_utils.rs`: lifter 코드를 제거하고 `mlxcel-core`에서 re-export.
- `src/models/`와 `src/audio/qwen3_omni_moe/` 아래 약 85개 설정 파일: `#[serde(remote = "Self")]`와 `impl_rope_lifting_deserialize!` 한 줄(보통 파일당 3줄).
- `src/models/mistral4.rs`, `src/models/exaone_moe.rs`, `src/models/exaone4.rs`, `src/models/jina_vlm.rs`: 위에서 설명한 flat 우선, `type` 처리, 폴백 변경.
- `src/loading/vlm_qwen.rs`, `src/lib/mlxcel-core/src/drafter/dflash/config.rs`: 호출 지점을 공유 lifter로 이동.
- `src/models/rope_parameters_lift_tests.rs`(신규, 417줄): 테이블 기반 테스트.

## 4. 검증

- `rope_parameters_lift_tests::every_flat_rope_theta_config_lifts_rope_parameters_and_flat_wins`는 89개 파싱 경로(끌어올린 struct 86개, Mistral 4, DFlash와 Jina의 `from_json` 경로)를 테이블로 순회한다. 중첩만 있으면 123456, flat 500000에 중첩이 있으면 500000인지 확인한다. RoPE가 아닌 필수 필드는 serde 자체 오류로 채우며 `rope_theta` / `rope_scaling`은 채우지 않는다.
- 추가 테스트: Llama 3 중첩 `llama3`가 `RopeScalingKind::Llama3`로 해석됨, Gemma 3 레이어 종류별, Phi `partial_rotary_factor`, DeepSeek-V4 `compress` YaRN(리뷰에서 추가, revert 실행은 하지 않음). lifter 단위 테스트는 `mlxcel-core`에 있다.
- 수정을 stash하고 테스트만 남기면 앞의 테스트 4개가 모두 실패하고, 89개 행 모두 실패를 보고한다.
- 실제 체크포인트: `~/models_archive/mlx/llama-3.2-1b-instruct-4bit`와 `~/models_archive/mlx/qwen3-30b-a3b-4bit`의 transformers-5 사본. `config.json`만 다시 쓰고(flat 키를 transformers 5.20.1이 해당 설정에 내보내는 것과 동일한 `rope_parameters` 블록으로 이동) 가중치는 심볼릭 링크로 연결했다. 명령: `MLXCEL_SDPA_DETERMINISTIC=1 gpu-lock run -- mlxcel generate -m <dir> -p "<CONTRIBUTING.md 앞 20000 바이트 + 질문>" -n 64 --temp 0 --show-reasoning`. 이 브랜치에서는 두 사본 모두 원본과 바이트 단위로 일치하고, 원본은 `main`과 일치한다. `main`(자기 일치 확인함)에서는 Llama 사본이 `and, and, ...`로, Qwen3 사본이 `. The. The. ...`로 퇴화한다.
- fmt, clippy(mlxcel과 mlxcel-core에 `--lib --tests -D warnings`), 영향받는 계열 테스트 필터(647개 테스트), 저장소 contract 검사가 로컬에서 통과한다.

## 5. 리뷰 중 변경

리뷰에서 CRITICAL 또는 HIGH 이슈는 발견되지 않았다. 커밋 `1684f151`에서 후속 수정 두 건을 적용했다.

- DeepSeek-V4 `main` / `compress` 레이아웃. 첫 버전은 이를 전부 `None`인 `rope_scaling`으로 만들었다.
- `rope_type` 스킴을 가리던 `"type": null` 키.

## 6. 미해결 사항

- 비전 인코더 설정(Pixtral, Llama 4 vision 등)과 `mlxcel-xla`. 이슈에 나열되지 않았다.
- DBRX는 transformers 5가 여전히 `rope_theta`를 두는 `attn_config`에서 끌어올리지만, 최상위 `rope_parameters`는 그곳으로 복사되지 않는다.

## 7. 변경 요약

| 항목 | 값 |
|------|----|
| 변경된 파일 | 96 |
| 추가된 줄 | 1134 |
| 삭제된 줄 | 116 |

| 커밋 | 메시지 |
|------|--------|
| `0681d42b` | fix(models): lift transformers 5 rope_parameters for every family |
| `1684f151` | fix(models): lift DeepSeek-V4 main/compress rope_parameters |
