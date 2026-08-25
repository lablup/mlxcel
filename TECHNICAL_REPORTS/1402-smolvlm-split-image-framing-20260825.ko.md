# 기술 보고서: PR #1402 - SmolVLM 분할 이미지 프레이밍 정합화

**작성일**: 2026-08-25
**작성자**: mlxcel 기여자
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: High

---

## 요약

PR #1402는 SmolVLM 및 Idefics3 이미지 분할을 체크포인트 설정과 upstream 프롬프트 프레이밍에 맞춘다. 실제 전처리 정책을 읽고, row-major 로컬 타일 뒤에 글로벌 썸네일을 배치하며, 토크나이저에서 유도한 행/열 마커 시퀀스를 생성하고, 프롬프트 placeholder와 비전 feature가 일대일로 대응하지 않으면 안전하게 실패한다.

---

## 1. 문제 정의

### 1.1 배경

SmolVLM 계열 체크포인트는 `preprocessor_config.json`에 이미지 분할 정책을 기록한다. 기존 로더는 주로 `processor_config.json`을 읽었고, 프로세서는 원본 이미지 크기로 crop을 선택했으며, 프롬프트 확장기는 모든 처리 이미지를 하나의 글로벌 블록으로 취급했다.

### 1.2 기존 문제점

- **설정 드리프트**: 체크포인트가 정의한 `do_image_splitting`, resize 제한, normalization 통계를 무시할 수 있었다.
- **기하 불일치**: 분할 crop이 기준 프로세서의 설정 기반 resize, clamp, 타일 배수 반올림, row-major 로컬 타일, 글로벌 썸네일 순서를 따르지 않았다.
- **프롬프트 불일치**: 로컬 타일에 `<row_r_col_c>` 프레이밍이 없어 텍스트 토큰 스트림이 전처리 feature 순서를 설명하지 못했다.
- **안전하지 않은 cardinality 경계**: 여분 또는 누락된 `<image>` placeholder가 비전 feature row와 다른 토큰 수로 masked feature merge에 도달할 수 있었다.

### 1.3 위험성

| 위험 | 영향도 | 수정 전 가능성 |
|------|--------|----------------|
| 분할 이미지 프롬프트가 잘못된 공간 마커와 feature를 연결 | High | High |
| 체크포인트별 normalization 및 크기 정책을 조용히 무시 | High | Medium |
| 잘못된 프롬프트/이미지 조합이 유효하지 않은 feature scatter를 유발 | High | Medium |

---

## 2. 변경 요약

### 2.1 체크포인트 기반 설정

로더는 분할 플래그, longest-edge 제한, 이미지 normalization 통계를 `preprocessor_config.json`에서 읽고 `processor_config.json`은 호환성 fallback으로만 유지한다. 타일당 이미지 토큰 수는 계속 비전 설정에서 가져온다.

### 2.2 기준 구현과 호환되는 타일링

프로세서는 설정된 resize와 clamp를 적용하고, resize된 크기를 타일 배수로 반올림한 뒤, 로컬 타일을 row-major 순서로 내보내고 글로벌 resize 이미지를 마지막에 추가한다. 각 이미지는 명시적인 `TileLayout { rows, cols }`를 반환하므로 프롬프트 생성기가 픽셀 수로 다시 추론하지 않고 정확한 로컬 타일 배치를 표현한다.

### 2.3 토크나이저 기반 프레이밍

SmolVLM 프롬프트 확장기는 special-token 자동 삽입을 끈 상태에서 체크포인트의 fake/image, 행/열, global 마커 문자열을 정확히 tokenize한다. 분할 이미지는 행마다 하나의 프레임 블록과 마지막 글로벌 블록을 받고, 단일 이미지는 global-only 프레이밍을 유지한다. Idefics2는 SmolVLM 분할 마커를 상속하지 않고 fake-only 단일 타일 계약으로 분리된다.

### 2.4 Fail-Closed Cardinality 검증

프롬프트 확장은 placeholder 수와 decode된 이미지/layout 수가 같아야 진행된다. 이미지 토큰 블록 크기는 checked arithmetic을 사용하고, 런타임은 확장된 이미지 토큰 위치 수와 인코딩 feature row 수가 같은지 확인하며, 0차원 이미지는 blank fallback 타일로 one-layout/one-tile 불변식을 유지한다. 이 검사는 사용자 제어 프롬프트 구조가 호환되지 않는 shape로 masked scatter에 도달하는 것을 막는다.

---

## 3. 기술적 선택과 그 이유

### 3.1 타일 레이아웃을 구조화된 메타데이터로 전달

**결정:** 처리된 픽셀 텐서와 함께 행·열 수를 반환한다.

**근거:** 프롬프트 프레이밍은 전처리 순서의 의미적 표현이다. 명시적 메타데이터를 사용하면 두 경로가 동기화되고 resize 및 반올림 후 취약한 기하 재추론을 피할 수 있다.

**트레이드오프:** 호출자와 통합 fixture가 더 풍부한 프로세서 결과를 채택해야 한다.

### 3.2 토큰 ID를 고정하지 않고 마커 문자열을 tokenize

**결정:** 각 모델 토크나이저가 생성한 마커 토큰 시퀀스를 캐시한다.

**근거:** 토큰 ID는 체크포인트 vocabulary 데이터이며 upstream 계약은 마커 문자열로 표현된다. `add_special_tokens = false`로 tokenize하면 하나의 vocabulary 배치를 가정하지 않고 관련 체크포인트 간 호환성을 유지할 수 있다.

**트레이드오프:** 모델 초기화 시 소량의 일회성 마커 tokenization을 수행하고 모델별 캐시를 보유한다.

### 3.3 확장 및 merge 경계에서 모두 검증

**결정:** placeholder 불일치를 조기에 거부하고 multimodal merge 직전에 token-feature cardinality를 다시 검사한다.

**근거:** 조기 검사는 명확한 입력 오류를 제공하고, 런타임 검사는 향후 프로세서 또는 프롬프트 회귀를 방어한다. malformed dimension에 대한 masked-scatter 동작에 의존하는 것보다 두 경계에서 방어하는 것이 안전하다.

---

## 4. 리뷰 및 품질 검토

### 4.1 구현 리뷰

통합 fixture를 체크포인트 normalization, `TileLayout`, 토크나이저 기반 프레이밍에 맞춘 뒤 구현 리뷰에서 해결되지 않은 correctness 이슈는 발견되지 않았다.

### 4.2 보안 및 성능 리뷰

보안 리뷰는 HIGH cardinality 불일치 한 건과 unchecked 이미지 토큰 산술 및 0차원 입력에 관한 MEDIUM 견고성 문제 두 건을 발견했다. 커밋 `ec5736feb`은 정확한 placeholder 검증, checked arithmetic, 런타임 feature 수 검증, blank-tile 불변식을 추가했다. 해결되지 않은 CRITICAL 또는 HIGH 보안·성능 이슈는 없다.

### 4.3 호환성

- **Breaking change**: CLI 또는 HTTP 인터페이스에는 없음.
- **새 의존성**: 없음.
- **동작 변경**: SmolVLM/Idefics3 분할 이미지는 체크포인트 정의 geometry와 marker framing을 사용하며, 잘못된 프롬프트/이미지 cardinality는 feature scatter에 도달하지 않고 거부된다.

---

## 5. 검증

- 최종 hardening 변경 뒤 `cargo test --workspace --profile test-fast --features metal,accelerate`가 통과했다.
- `cargo clippy --workspace --all-targets -- -D warnings`가 통과했다.
- `cargo fmt --all -- --check`와 `git diff --check`가 통과했다.
- 집중 `smolvlm_parity` 통합 suite의 finite-logit, processor, normalization, detection, prompt-token 검증 6개가 모두 통과했다.
- 실제 로컬 Idefics3 8B 4-bit 체크포인트를 Metal에서 실행해 3x4 로컬 분할과 글로벌 썸네일에서 2,197개 이미지 토큰(`13 × 169`), 단일 256x256 이미지에서 169개 토큰을 확인했다. 단일 타일 실행의 첫 greedy 토큰은 기존 release binary와 일치했다.
- 로컬 토크나이저 기준 검사에서도 3x4 분할 프롬프트의 `<image>` 토큰 ID 2,197개와 단일 이미지의 169개를 독립적으로 확인했다.
- Full mlx-vlm token-exact 생성은 해당 `Idefics3ImageProcessor` 선언에도 로컬 체크포인트를 거부하는 `AutoProcessor` 문제로 시작하지 못했다. 이 외부 reference-loader 제한은 실제 체크포인트 mlxcel 실행을 막지 않았다.

---

## 6. 변경 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 7 |
| 추가 줄 | 876 |
| 삭제 줄 | 235 |
| 구현 커밋 | 3 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `b64e0ce2a` | fix | SmolVLM 분할 이미지 프레이밍 정합화 |
| `18ebdb85e` | test | 분할 프레이밍 SmolVLM parity coverage 갱신 |
| `ec5736feb` | fix | SmolVLM 이미지 토큰 cardinality 보호 |

---

## 7. 후속 고려 사항

- 로컬 mlx-vlm/Transformers 프로세서 스택이 체크포인트 메타데이터를 인식하면 token-exact 기준 생성을 다시 실행한다.
- 추가 VLM 계열에 분할을 확장할 때 텐서 수로 프롬프트 기하를 추론하지 말고 명시적인 타일 레이아웃 전달을 유지한다.
- 체크포인트 근거로 공용 마커 계약이 확인되기 전까지 Idefics2와 SmolVLM 프레이밍 경로를 분리한다.

---

## 참고

- 이슈 #1364: 체크포인트 분할 정책 및 행/열 프레이밍 요구사항
- PR #1402: 구현, parity 갱신, cardinality hardening
- `docs/supported-models.md`: SmolVLM/Idefics3 분할 이미지 동작
