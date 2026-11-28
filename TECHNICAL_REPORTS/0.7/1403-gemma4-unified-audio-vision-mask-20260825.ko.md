# 기술 보고서: PR #1403 - 오디오 포함 Gemma4 비전 마스크 유지

**작성일**: 2026-08-25
**작성자**: mlxcel 기여자
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: High

---

## 요약

PR #1403은 오디오 토큰이 같은 프롬프트에 있을 때도 Gemma 4 Unified의 연속 이미지·비디오 토큰 구간에 blockwise bidirectional attention을 유지한다. 두 마스크 생성 경로에서 잘못된 오디오 존재 gate를 제거하고, 오디오는 모든 비전 블록 밖에 유지하며, 비전 위치만 forward attention을 얻는 회귀 테스트를 추가했다.

---

## 1. 문제 정의

### 1.1 배경

Gemma 4 Unified 체크포인트는 `use_bidirectional_attention: "vision"`을 선언할 수 있다. Prefill 동안 하나의 연속 이미지 또는 비디오 토큰 구간 안의 위치는 전체 구간에 attend해야 하며, 텍스트와 오디오 row는 기존 causal 또는 windowed mask를 유지해야 한다.

### 1.2 기존 문제점

- **Host-side 정책 오류**: 유효한 비전 구간이 있어도 오디오 토큰이 하나라도 있으면 `compute_vision_block_ids`가 `None`을 반환했다.
- **Graph-side 정책 오류**: embeddings-driven prefill 경로도 MLX 연산으로 비전 위치를 유도한 뒤 동일한 오디오 count gate를 반복했다.
- **실제 도달 가능한 품질 저하**: Gemma 4 Unified는 이미 이미지와 오디오 결합 요청을 받으므로 운영 프롬프트의 이미지 구간이 조용히 완전 causal로 실행될 수 있었다.
- **잘못된 회귀 고정**: 단위 테스트가 혼합 프롬프트에 대해 저하된 `None` 결과를 명시적으로 요구했다.

### 1.3 위험성

| 위험 | 영향도 | 수정 전 가능성 |
|------|--------|----------------|
| 오디오를 함께 제공할 때만 이미지 이해 품질 저하 | High | High |
| Host와 embeddings-driven prefill 마스크가 체크포인트 의도와 불일치 | High | High |
| 향후 리팩터링에서 잘못된 mixed-modality gate를 유지 | Medium | High |

---

## 2. 변경 요약

### 2.1 오디오와 독립적인 비전 블록

Host helper는 이제 활성화된 체크포인트 정책, 토큰 두 개 이상의 prefill sequence, 최소 하나의 이미지 또는 비디오 토큰만 요구한다. 각 연속 비전 run에는 0 이상의 block id가 부여되고 오디오를 포함한 나머지 모든 위치는 `-1`을 유지한다.

### 2.2 일치하는 MLX Graph 경로

`Gemma4UnifiedModel::block_ids_array_for`는 더 이상 오디오 존재 scalar를 만들거나 읽지 않는다. 비전 존재 검사는 유지하고 기존 vision mask, block-start 탐지, 누적 번호 부여, non-vision `-1` 할당을 그대로 사용하므로 graph 경로와 host helper가 동일한 계약을 갖는다.

### 2.3 정확한 마스크 회귀 검증

오디오가 overlay를 끈다고 가정하던 테스트는 이제 `Some([-1, 0, 0, -1, -1])`을 기대한다. 새 additive-mask 테스트는 `[BOI, image, image, EOI, audio, audio, text]`를 구성해 첫 이미지 토큰이 두 번째 이미지 토큰으로 forward attend할 수 있지만 첫 오디오 토큰은 두 번째 오디오 토큰으로 forward attend할 수 없음을 확인한다.

### 2.4 문서화

지원 모델 항목은 오디오 토큰이 Gemma 4 Unified 비전/비디오 블록 밖에 있고 causal row를 유지한다고 명시한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 비전 Run만으로 Overlay 정의

**결정:** 오디오 존재 여부는 overlay gate에 참여하지 않는다.

**근거:** Overlay 관계는 `같은 0 이상의 vision block`이다. 오디오 위치는 항상 `-1`이므로 same-block match에 들어갈 수 없고 별도의 비활성화 규칙이 필요 없다.

**트레이드오프:** 혼합 프롬프트는 더 저렴한 완전 causal fallback 대신 이미지-only 프롬프트와 같은 비전 overlay mask를 materialize하며, 이는 체크포인트 정확성을 위해 필요하다.

### 3.2 공유 Helper 의미 유지

**결정:** Gemma 4 Unified 호출부 예외를 추가하지 않고 공유 block-id helper를 수정한다.

**근거:** DiffusionGemma도 이 helper를 사용하지만 `audio: -1`을 전달하므로 오디오 gate 제거가 해당 입력을 바꾸지 않는다. 하나의 계약이 host와 graph 마스크 정책의 드리프트를 방지한다.

---

## 4. 리뷰 및 품질 검토

### 4.1 구현 리뷰

구현 리뷰에서 해결되지 않은 correctness 이슈는 발견되지 않았다. Host helper, MLX graph 연산, additive-mask 의미, DiffusionGemma 재사용, 문서를 집중 검토했다.

### 4.2 보안 및 성능 리뷰

남은 CRITICAL, HIGH, MEDIUM, LOW 보안·성능 이슈는 없다. 이 변경은 graph 경로에서 scalar reduction/readback 하나를 제거하며 shape 기반 할당 또는 비신뢰 인덱싱을 추가하지 않는다. 오디오 토큰은 block id `-1`을 통해 구조적으로 제외된다.

### 4.3 호환성

- **Breaking change**: CLI 또는 HTTP 인터페이스에는 없음.
- **새 의존성**: 없음.
- **동작 변경**: 이미지/비디오와 오디오를 결합한 프롬프트가 완전 causal 비전 row로 fallback하지 않고 체크포인트의 vision-bidirectional 정책을 적용한다.

---

## 5. 검증

- 최종 리뷰 뒤 `cargo test --workspace --profile test-fast --features metal,accelerate`가 통과했다.
- `cargo clippy --workspace --all-targets -- -D warnings`가 통과했다.
- `cargo fmt --all -- --check`와 `git diff --check`가 통과했다.
- 집중 Gemma 4 Unified 마스크 선택은 10개 테스트, 더 넓은 Gemma 4 Unified 선택은 50개 테스트, DiffusionGemma 선택은 39개 테스트가 통과했고 2개가 ignore됐다.
- 로컬 upstream 회귀 테스트 `test_gemma4_unified_audio_tokens_keep_vision_overlay`가 통과해 mixed-audio 마스크 관계를 독립적으로 확인했다.
- 실제 로컬 Gemma 4 Unified 12B 4-bit 체크포인트를 Metal에서 실행해 이미지+오디오 및 이미지-only greedy 실행이 유한하고 자연스러운 출력을 내는 것을 확인했다. 16 kHz 혼합 입력은 로컬 reference와 같은 324-token 프롬프트 길이로 확장됐다.
- 실제 생성의 완전한 토큰 일치는 성립하지 않았다. mlxcel과 로컬 reference는 공통 시작 토큰 시퀀스와 같은 의미의 답을 만들었지만 이후 묘사 어휘가 달라졌다. 별도로 링크된 변경 전·후 image-only 바이너리도 image-only 마스크 값이 변하지 않았는데 이후 greedy 어휘가 달라, 최적화 빌드 수치에 exact real-output 비교가 민감함을 확인했다.

---

## 6. 변경 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 4 |
| 추가 줄 | 44 |
| 삭제 줄 | 22 |
| 구현 커밋 | 1 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `8dc5e0169` | fix | 오디오 포함 Gemma4 Unified 비전 overlay 유지 |

---

## 7. 후속 고려 사항

- 독립적으로 링크된 최적화 바이너리 사이에서도 greedy 토큰 일치를 강제해야 한다면 안정적인 첫 logit 또는 mask tensor 실체크포인트 parity harness를 추가한다.
- 향후 multimodal mask gate는 같은 프롬프트의 무관한 modality가 아니라 overlay에 참여하는 modality만 기준으로 삼는다.
- Gemma 4 Unified batching 또는 speculative prefill 경로를 확장할 때 오디오 `-1` 불변식을 유지한다.

---

## 참고

- 이슈 #1344: mixed audio가 비전 overlay를 잘못 비활성화
- PR #1403: 회귀 검증을 포함한 host 및 graph 마스크 수정
- `docs/supported-models.md`: Gemma 4 Unified multimodal 마스크 동작
