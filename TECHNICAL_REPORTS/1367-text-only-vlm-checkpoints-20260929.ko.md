# 기술 보고서: 비전 타워가 제거된 VLM 체크포인트를 텍스트 전용 경로로 라우팅 (#1367)

**날짜**: 2026-09-29

**상태**: 구현 및 로컬 검증 완료, 머지 대기.

**언어**: Rust

**위험도**: 낮음

## 요약

커뮤니티 체크포인트 중에는 VLM `model_type`은 유지하되 비전 타워를 제거한 것이 늘고 있다. 기존에는 `vision_config`를 생략한 체크포인트만, 그것도 네 개 패밀리에서만 로드되었다. 이제 detection은 설정, 플래그, 가중치 이름 세 가지 신호로 텍스트 전용 여부를 판단하며, 텍스트용 `ModelType`을 재사용할 수 없는 Qwen-VL 네 패밀리는 비전 인코더 없이 로드되고 미디어 요청을 명시적 오류로 거부한다.

## 1. 문제

`has_vision_config`는 `vision_config` 키의 존재만 확인했기 때문에 `"vision_config": {}`나 `null`이 VLM 로더로 가서 서브 설정 파싱이나 첫 `vision_tower.*` 텐서에서 실패했다. `language_model_only: true`는 바로 그 제거된 빌드에 대해서도 명시적 오류였다. Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE에는 텍스트 전용 경로가 없었다.

## 2. 변경 요약

- `detection.rs`: 부재, `null`, `{}`, `language_model_only: true`를 텍스트 전용으로 처리한다(플래그는 비전 가중치가 있어도 우선하며 stderr에 한 줄 안내를 출력). `vlm_has_vision(config, path)`는 `VLM_VISION_WEIGHT_PREFIXES`로 가중치 이름을 검사한다(인덱스 우선, 없으면 safetensors 헤더). `qwen3_5`, `qwen3_5_moe`, `gemma3`가 새 `ModelDetectionProbes::vlm_has_vision_weights`를 사용하며, 카탈로그 프로브는 제한된 인덱스로 답하고 읽을 수 없으면 VLM 판정을 유지한다.
- `vlm_qwen.rs`: 네 Qwen-VL 로더가 `vlm_has_vision`을 호출하고, false면 비전 설정 필수 조건과 인코더 생성을 건너뛴다.
- `vision/qwen*_vl*.rs`: `vision_encoder`가 `Option`이 되고 `text_only_path`가 추가되었다. `QwenVlRuntime::text_only_source`와 `LoadedModel::has_vision_tower`로 노출하며, `compute_qwen_vl_media_embeddings`가 `model <path> was loaded without a vision tower (text-only checkpoint)` 오류를 반환한다.
- `qwen3_5.rs`: `language_model_only: true`는 더 이상 오류가 아니며, 불리언이 아닌 값은 여전히 오류다.

## 3. 기술적 결정

- 가중치 검사는 설정상 VLM인 모델을 텍스트로 낮추는 방향으로만 작동한다. 가중치를 읽을 수 없으면 설정 판정을 유지하므로 기존에 동작하던 경로는 바뀌지 않는다.
- `llama4`, `mistral3`, `gemma3n`은 설정 규칙만 적용한다. 목록에 없는 접두사에 타워가 있으면 텍스트로 잘못 라우팅되어 현재보다 나쁘게 실패할 수 있다.
- Qwen-VL의 `ModelType`은 그대로 두고 로더가 같은 `vlm_has_vision` 함수를 호출해 판정을 전달한다. 새 enum 변형이 없어 `detection.rs` 변경이 최소화된다.
- 비전 래퍼의 `expect`는 런타임 검사가 먼저 실행되므로 도달할 수 없다.

## 4. 검증

- 단위: `vlm_text_only_detection_tests`(빈/null `vision_config`, 플래그, 인덱스 검사, 모든 접두사, 단일 파일 헤더, 읽을 수 없는 가중치, Qwen-VL 타입 유지)와 `vlm_qwen_text_only_tests`(합성 `qwen3_vl` 체크포인트가 `vision_encoder: None`으로 로드되고 이미지를 명시적으로 거부).
- 실제 체크포인트 `leonsarmiento/ThinkingCap-Qwen3.6-27B-3bit-mlx`: 원본, `vision_config: {}`, `language_model_only: true`, 전체 `vision_config`, 전체 `vision_config`와 플래그 조합 모두 동일한 greedy 출력(64 토큰)을 냈다. `--image` 요청은 패닉 없이 오류를 반환한다.

## 5. 미완료

`mlxcel list`의 기능 표시는 `model_type` 기준의 정적 값이라 텍스트 전용으로 로드된 Qwen2/2.5/3-VL도 VLM으로 표시된다. 강제는 런타임 거부가 담당한다. LLaVA, Idefics, Molmo, Pixtral의 텍스트 전용 경로는 범위 밖이다.
