# 기술 보고서: PR #2133 - fix(youtu_vl): 템플릿 이미지 placeholder를 병합 feature 수만큼 확장

**날짜**: 2026-10-06
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (CUDA에서 실제 체크포인트로 CLI와 mlxcel-server 모두 검증)
**언어**: Rust
**위험도**: 낮음 (한 모델 계열의 프롬프트 준비 경로만 변경, 새 개수 검사는 조용한 잘림을 요청 오류로 바꿈)

---

## 요약

#1610이 패치 방출 순서를 고친 뒤에도 Youtu-VL은 세 도형 fixture 두 개를 모두 "원 하나"로 설명했고, 단색 주황 대조 이미지만 올바르게 읽었습니다. 올바른 fixture가 유일하게 단일 attention window 크기였기 때문에 #1618은 windowed attention 경로를 의심했습니다. 실제 원인은 vision tower 앞단에 있었습니다. 채팅 템플릿은 이미지마다 `<|image_pad|>` 하나를 렌더링하는데 mlxcel은 이를 병합 feature 수만큼 확장하지 않았고, masked scatter가 첫 번째 feature 하나만 프롬프트에 넣었습니다. PR #2133은 체크포인트의 processor와 같은 방식으로 placeholder를 확장하고, 개수가 맞지 않으면 오류로 처리합니다. 세 fixture 모두에서 CLI와 서버 출력이 transformers greedy 출력과 단어 단위로 일치합니다.

---

## 1. 문제 정의

### 1.1 배경

체크포인트의 `processing_youtu_vl.py`에 있는 `YoutuVLProcessor.__call__`은 렌더링된 텍스트의 각 `<|image_pad|>`를 병합 vision feature 하나당 하나씩, 즉 `h * w / 4`개로 바꿉니다. mlxcel의 대응 함수 `insert_youtu_vl_image_tokens`는 placeholder 없이 이미지만 넘기는 호출자를 위해 작성되어 BOS 뒤에 framing된 토큰 열을 삽입합니다. 이 함수는 프롬프트에 이미지 토큰이 하나라도 있으면 이미 확장된 것으로 보고 바로 반환했습니다.

### 1.2 기존 문제

채팅 템플릿은 항상 placeholder를 렌더링하므로 이 조건이 모든 템플릿 요청에서 걸렸습니다. 프롬프트에는 이미지 토큰이 하나(`128262, 128264, 128263`)뿐이었고 feature는 49, 121, 196개였습니다. `merge_llava`의 masked scatter는 placeholder를 순서대로 채우고 남는 것은 버리므로, 언어 모델은 왼쪽 위 병합 패치 하나만 보았습니다.

### 1.3 window 결함처럼 보인 이유

균일한 이미지는 모든 병합 토큰이 같은 내용을 담으므로 224 주황 fixture는 남은 토큰 하나로도 올바르게 읽혔습니다. 두 도형 fixture는 모서리가 모두 밝은 회색이라 모델은 "단색 배경 위의 원 하나"를 두 가지 극성으로 답했습니다. 격자 크기와 window 수가 정답 여부와 상관관계를 보인 것은 균일한 fixture가 마침 작은 이미지였기 때문입니다.

---

## 2. 기술 검토

### 2.1 단계별 비교

transformers 4.56.0 oracle(체크포인트 원격 코드, CPU f32)로 참조 processor 출력과 tower+merger 출력을 만들었습니다. mlxcel processor는 224와 448에서 3e-8 이내로 일치했고 336은 Lanczos와 bilinear 리샘플링 차이뿐이었습니다. 참조 픽셀을 mlxcel bf16 tower에 넣었을 때 병합 토큰별 cosine 평균은 14x14, 22x22, 28x28 패치 격자에서 각각 0.9999, 0.998, 0.997이었습니다. 이로써 이슈가 지목한 `get_window_index`, `cu_window_seqlens` 패딩과 중복 제거, full-attention 경계, window 역순열이 모두 배제됩니다.

### 2.2 결함 위치 특정

참조 구현은 bf16에서도 정답을 생성했고 vision feature도 일치했으므로 차이는 tower 이후에 있었습니다. 프롬프트 id를 덤프하자 참조 구현에서 196개인 이미지 토큰이 mlxcel에는 하나뿐이었습니다.

---

## 3. 기술적 결정

### 3.1 processor의 치환 루프를 그대로 따름

이미지당 placeholder 하나는 템플릿의 시작/끝 framing을 유지한 채 그 자리에서 확장합니다. feature당 토큰이 이미 있는 프롬프트는 그대로 두며, 토큰 열이 하나씩인 격자를 두 번 확장하지 않도록 이 조건을 먼저 검사합니다. placeholder가 없으면 기존처럼 BOS 뒤에 삽입합니다. 그 밖의 개수는 추측하지 않고 오류로 처리합니다.

### 3.2 placeholder와 feature 개수 불일치 시 실패

upstream은 이미지 토큰과 feature 수가 다르면 예외를 냅니다. 이제 `YoutuVLModel::get_input_embeddings`도 같은 검사를 하므로 이런 결함은 패치 하나에 대한 유창한 설명 대신 요청 오류로 드러납니다.

---

## 4. 검증

| fixture | 이전 | 이후 (CLI와 서버) |
|---|---|---|
| 224 주황 | "a solid, uniform orange color" | 변화 없음 |
| 336 도형 | "a single white circle on a black background" | 빨간 사각형, 파란 원, 초록 삼각형 (transformers와 동일) |
| 448 도형 | "a single, solid black circle on a plain white background" | "- Red square - Blue circle - Green triangle" (transformers와 동일) |

`tests/youtu_vl_parity.rs`에서 실패로 알려졌던 두 테스트가 실제 체크포인트에서 통과합니다. 서버의 `prompt_tokens`는 82, 154, 229로 참조 processor의 토큰 수와 같습니다. 새 단위 테스트는 28x28 격자 확장, 다중 이미지 순서, 불일치 오류, 다중 window 격자에서 참조값과의 `get_window_index` 일치를 다룹니다.

---

## 5. 변경 요약

- `src/multimodal/youtu_vl_prompt.rs`, `youtu_vl_prompt_tests.rs`: placeholder 확장과 테스트.
- `src/vision/youtu_vl.rs`: scatter 전 개수 검사.
- `src/multimodal/vlm_runtime.rs`: 두 오류를 요청 오류로 전파.
- `src/vision/encoders/youtu_vl_tests.rs`: window index 참조 테스트.
- `tests/youtu_vl_parity.rs`: ignore 사유 정리, 주석 수정.

관련: #1618 (이 PR로 종료), #1610, #1600, #1611.

---

## 6. 후속 조치

### 일반화할 교훈

단색 fixture는 feature가 프롬프트 슬롯에 도달하는 방식의 결함을 잡지 못합니다. feature 하나가 이미지 전체를 대표하기 때문입니다. 내용 테스트에는 공간적으로 다양한 fixture가 필요하고, VLM 포팅은 잘림을 허용하는 scatter에 기대지 말고 placeholder 수와 feature 수가 같은지 확인해야 합니다.

### 남은 항목

processor의 `smart_resize`는 가장자리를 32의 배수로 반올림하지만 참조 구현은 올림하므로, 이번 fixture가 아닌 일부 입력 크기에서는 패치 격자가 달라집니다. #1611의 전제(`max_num_patches` = 256으로 제한)는 `YoutuVLProcessor.__call__`이 `max_image_patches=36864`를 넘겨 preprocessor 설정을 덮어쓴다는 사실과 모순됩니다.
