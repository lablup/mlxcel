# 기술 보고서: PR #1400 - 모델 소유 캐시에 KV 모드 적용

**작성일**: 2026-08-25
**작성자**: mlxcel 기여자
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: High

---

## 요약

PR #1400은 이질적인 시퀀스 상태를 모델 내부에서 할당하는 모델군에도 해석 완료된 KV 캐시 정책을 전달한다. CLI/서버 관측 정보, 캐시 통계, 스냅샷 거부 정책, dense 프롬프트 캐시 채택을 실제 활성 모드와 일치시켜 Gemma, Qwen, LFM2, Llama 4, AFMoE 및 관련 VLM 래퍼에서 발생하던 조용한 설정 무효화를 제거했다.

---

## 1. 문제 정의

### 1.1 배경

CLI 생성기와 서버 캐시 풀은 이미 `--kv-cache-mode`, Boundary-V, `--kv-bits`를 계층별 정책으로 해석하고 있었지만, 이 정책은 `LanguageModel::forward`에 전달되는 외부 `Vec<KVCache>`에만 적용됐다. 모델 소유 계열은 순환, 합성곱, 슬라이딩 윈도우 등 서로 다른 상태를 함께 관리해야 하므로 해당 균질 슬라이스를 사용하지 않고 내부 attention 캐시를 FP16 기본값으로 생성했다.

### 1.2 기존 문제점

- **조용한 무효화**: 운영자가 Int8 또는 Turbo 모드를 요청해도 모델 소유 계열은 오류 없이 FP16 attention 캐시를 계속 할당했다.
- **잘못된 관측 정보**: 배너와 통계가 batch KV 양자화가 선택한 실제 모드가 아니라 요청 모드 또는 레거시 모드를 표시할 수 있었다.
- **안전하지 않은 재사용 경계**: 일반 key/value만 복사하는 스냅샷 직렬화가 양자화 sidecar를 누락할 수 있었고, dense 프롬프트 캐시 채택은 계층별 KV 모드 불일치를 거부하지 않았다.
- **불완전한 모델군 지원**: 문서화된 대칭 Turbo4 allowlist의 Qwen 계열도 내부 캐시에 해석 완료 정책을 적용하지 못했다.

### 1.3 위험성

| 위험 | 영향도 | 수정 전 가능성 |
|------|--------|----------------|
| 운영 메모리 사용량이 공지된 설정과 불일치 | High | High |
| 프롬프트 또는 스냅샷 재사용이 호환되지 않는 KV 표현을 혼합 | High | Medium |
| 부정확한 캐시 통계를 기반으로 성능 튜닝 | Medium | High |

---

## 2. 변경 요약

### 2.1 정책 주입

`LanguageModel`은 이제 해석 완료된 계층별 KV 모드 테이블을 설정하고 조회하는 훅을 제공한다. 공용 `KvCacheLayerModes` 저장소가 모델 소유 시퀀스 상태 옆에 테이블을 보관하고, `LoadedModel`과 해당 VLM 래퍼가 기반 텍스트 모델로 훅을 위임한다.

두 실행 경로 모두 effective-mode 해석 뒤 정책을 주입한다. 오프라인 생성에서는 `CxxGenerator`, 서버 추론에서는 batch scheduler가 이를 담당한다. Boundary-V와 batch KV 양자화는 기존 해석 로직을 계속 사용하므로 새로운 requested-mode 경로를 만들지 않고 하나의 진실 원천을 확장한다.

### 2.2 모델군 적용 범위

Gemma 3, AFMoE, Gemma 4, Qwen 3.5, Qwen3-Next, Bailing MoE Linear, LFM2, Llama 4 일반 attention 생성자는 계층 인덱스에 맞는 설정 모드를 선택한다. 순환, 합성곱, gated-delta, Llama 4 `ChunkedKVCache` 상태는 FP16을 유지하며, Llama 4는 비-FP16 정책이 지원되지 않는 chunked 계층에 도달하면 한 번 경고한다.

### 2.3 재사용 안전성

Dense detached 프롬프트 캐시는 모든 계층에서 해석 완료 모드와 비교된다. 불일치는 다른 표현을 사용하는 live scheduler에 채택되지 않고 명시적인 `kv_mode_mismatch` 사유와 메트릭으로 거부된다.

모델 소유 exact-prefix 스냅샷 경로는 필요한 표현을 보존할 수 없는 비-FP16 상태를 거부한다. 이 PR은 Int8 또는 Turbo sidecar를 잃은 채 key/value만 복원하는 대신 명시적으로 거부하는 보수적 정책을 선택했다.

### 2.4 관측 가능성

CLI 배너와 서버 시작 로그는 effective mode가 적용된 계층 수를 보고한다. `/v1/cache/stats`는 `kv_cache_mode_effective`를 제공하며, 리뷰 수정으로 `--kv-bits`가 활성화된 경우 레거시 FP16 필드가 아니라 `BatchKvQuantConfig::base_mode()`를 우선 사용한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 원시 플래그가 아니라 해석 완료 테이블 주입

**결정:** 모델 계층 인덱스에 대응하는 해석 완료 모드 벡터를 저장한다.

**근거:** 단일 원시 플래그는 Boundary-V 또는 계층별 batch KV 정책을 표현할 수 없으며 직전 이슈에서 제거한 requested/effective 불일치를 다시 만들 수 있다. 테이블을 사용하면 이질적 모델 래퍼가 attention 계층에만 정책을 적용하고 순환 상태는 그대로 유지할 수 있다.

**트레이드오프:** 모델이 작은 설정 상태를 보유하며, 새 VLM 계열을 추가할 때 래퍼 위임을 빠뜨리지 않아야 한다.

### 3.2 지원되지 않는 스냅샷은 거부

**결정:** 비-FP16 모델 소유 스냅샷을 부분 직렬화하지 않고 fail-closed 한다.

**근거:** 완전한 Turbo 또는 Int8 스냅샷에는 packed tensor, norm, scale, seed, offset, threshold가 필요하다. 일반 key/value만 복사하면 유효해 보이지만 표현이 다른 캐시가 만들어진다. 명시적 거부로 정확성을 지키고 전체 sidecar 직렬화는 별도 변경으로 남긴다.

**트레이드오프:** 일부 비-FP16 exact-prefix 캐시 hit가 cold prefill로 대체된다.

### 3.3 Chunked KV 상태는 FP16 유지

**결정:** Llama 4 일반 attention 캐시만 양자화하고 지원되지 않는 `ChunkedKVCache` 계층에는 경고한다.

**근거:** Chunked 캐시는 아직 양자화 sidecar 저장소를 구현하지 않았다. 부분 적용을 명시하는 것이 완전 지원을 조용히 주장하거나 이 이슈에서 캐시 형식을 바꾸는 것보다 안전하다.

---

## 4. 리뷰 및 품질 검토

### 4.1 구현 리뷰

구현 리뷰에서 HIGH 이슈 한 건을 발견했다. 서버 batch KV 양자화가 활성화된 경우에도 캐시 통계가 레거시 effective-mode 필드를 읽고 있었다. 커밋 `01773044a`는 route가 batch 양자화 base mode를 우선 사용하도록 수정하고 Int8, Turbo, 레거시 fallback 테스트를 추가했다.

### 4.2 보안 및 성능 리뷰

해결되지 않은 CRITICAL 또는 HIGH 보안/성능 이슈는 없었다. 보안 리뷰는 새 재사용 경계가 fail-closed 하며, 모드 테이블이 검증된 런타임 설정에서 유도되고, 비신뢰 입력으로 해석된 계층 수를 넘어 인덱싱하지 않음을 확인했다.

### 4.3 호환성

- **Breaking change**: CLI 플래그와 공개 HTTP 요청 형식에는 없음.
- **새 의존성**: 없음.
- **동작 변경**: 기존 비-FP16 플래그가 모델 소유 attention 캐시에 실제 적용되며, 지원되지 않는 모델 소유 스냅샷 재사용은 불완전 상태를 허용하는 대신 cold prefill로 전환될 수 있다.

---

## 5. 검증

- 최종 리뷰 변경 뒤 `cargo test --workspace --profile test-fast --features metal,accelerate`가 통과했다.
- `cargo clippy --workspace --all-targets -- -D warnings`가 통과했다.
- `cargo fmt --all -- --check`와 `git diff --check`가 통과했다.
- 생성기 모드 전달, Qwen3-Next 스냅샷 모드 불일치 거부, dense 프롬프트 캐시 KV 모드 거부, effective cache-stat 보고에 대한 집중 테스트가 통과했다.
- 실제 Gemma 3 모델로 Metal에서 FP16과 Int8 생성을 검증했다. 두 배너 모두 `applied to 26 of 26 layers`를 보고했고, 출력은 유창하고 유한했으며, greedy 토큰 스트림이 달라 Int8이 내부 attention 캐시에 도달했음을 확인했다.

---

## 6. 변경 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 26 |
| 추가 줄 | 1,009 |
| 삭제 줄 | 204 |
| 구현 커밋 | 2 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `b3536bff9` | fix | 모델 소유 캐시에 KV 모드 적용 |
| `01773044a` | fix | 캐시 통계에 batch KV 모드 보고 |

---

## 7. 후속 고려 사항

- 캐시 hit 성능이 추가 포맷 복잡성을 정당화하면 모델 소유 exact-prefix 스냅샷에 완전한 양자화 sidecar 직렬화를 구현한다.
- Llama 4 `ChunkedKVCache`와 ring-sliding 캐시의 양자화 표현은 별도 범위에서 추가한다.
- `mlxcel_prompt_cache_reject_total{reason="kv_mode_mismatch"}`를 관찰해 의도된 설정 변경과 예기치 않은 정책 드리프트를 구분한다.
- 새 VLM 래퍼 또는 모델 소유 계열을 추가할 때 `LanguageModel` KV 모드 훅 위임을 유지한다.

---

## 참고

- 이슈 #1330: 모델 소유 KV 캐시 모드 무효화 및 재사용 안전성 요구사항
- PR #1400: 최종 구현 및 리뷰 수정
- `docs/turbo-kv-cache.md`: 지원 모드, 모델 소유 동작, 현재 예외
