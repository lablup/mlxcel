# 기술 보고서: PR #1606 - feat: add arch JSON recipes registry

**작성일**: 2026-09-03
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (머지된 PR에 Rust, CLI, clippy, registry rebuild 검증이 기록돼 있고, 후속 fix 커밋 두 개가 레지스트리 계약 불일치와 아티팩트 손상 위험까지 마감했다)
**언어**: Rust, Markdown, JSON
**위험도**: Medium (`mlxcel arch --json`이 하위 소비자 계약이 되고 큰 생성 산출물을 커밋하므로, 단순 CLI 추가보다 스키마 드리프트와 재생성 안전성이 더 중요하다)

---

## 요약

레시피 사이트는 모델 패밀리 인벤토리를 안정적으로 읽을 수 있는 기계 판독 형식이 필요했지만, `mlxcel arch`는 사람이 읽는 카탈로그만 제공했다. 그래서 하위 자동화는 runtime, modality, backend, distributed, drafter, KV-cache 지원 여부를 prose 파싱이나 내부 enum 재구현 없이 얻을 수 없었다.

PR #1606은 `mlxcel arch --json`을 그 계약으로 추가하고, `ALL_MODEL_TYPES`의 각 variant마다 하나의 registry entry를 유도하며, 첫 `recipes/registry/0.6.0.json` 스냅샷과 `CURRENT`를 커밋한다. 이어진 두 개의 fix 커밋은 광고되는 KV 모드를 실제 MLA resolver와 맞추고, `make recipes-registry`가 아티팩트를 원자적으로 갱신하도록 보강했다.

---

## 1. 문제 정의

### 1.1 배경

저장소에는 이미 큰 모델 taxonomy가 있었지만 Rust enum과 capability helper 안에만 존재했다. 이는 CLI 렌더링에는 충분했지만, `recipes.vllm.ai` 같은 정적 레지스트리 패턴을 따르는 recipes 워크플로에는 부족했다. 하위 빌더는 직접 읽을 수 있는 버전드 JSON이 필요하다.

### 1.2 기존 문제점

- `mlxcel arch`는 레시피 빌더나 정적 사이트 생성기가 아니라 사람을 위한 출력이었다.
- 정식 아키텍처 정보가 `ALL_MODEL_TYPES`, family helper, distributed flag, KV capability 로직에 흩어져 있어 외부 도구가 기대할 단일 소스가 없었다.
- 생성 타깃이 추적 중인 파일에 직접 덮어쓰면, JSON 생성이 중간에 실패할 때 부분적으로 깨진 결과를 남길 수 있었다.

### 1.3 방치 시 위험

안정적인 exported registry가 없으면 하위 소비자는 전부 터미널 출력 파싱이나 별도 모델 매트릭스 재구현에 의존하게 된다. 둘 다 빠르게 드리프트한다. 더 나쁜 점은 rebuild가 실패했을 때 committed registry artifact가 잘린 채 남을 수 있다는 것이다. 그러면 자동화는 `CURRENT`가 가리키는 스냅샷을 정상이라고 오해한 채 손상된 파일을 읽게 된다.

---

## 2. 기술적 검토

### 2.1 레지스트리 계약

`src/models/registry.rs`는 recipes registry 전용 직렬화 레이어를 도입한다.

- `ArchitectureRegistry`는 `mlxcel_version`과 `families` 배열을 담는다.
- `ArchitectureFamily`는 안정적인 `id`, 표시 이름, category, detection key, runtime, modality, output kind, backend 지원 상태, tensor/pipeline-parallel 플래그, drafter 지원, KV mode를 담는다.
- 작은 enum들(`Runtime`, `Modality`, `OutputKind`, `BackendStatus`, `Drafter`, `KvMode`)은 전부 `snake_case`로 직렬화돼 하위 정적 생성기가 예측 가능한 JSON을 받는다.

핵심은 registry 데이터가 별도 수작업 테이블이 아니라 `ALL_MODEL_TYPES`에서 유도된다는 점이다. 덕분에 패밀리 커버리지의 소유권은 하나로 유지하면서, 외부가 소비할 수 있는 스냅샷만 별도로 내보낼 수 있다.

### 2.2 CLI 표면

`src/main.rs`는 기존 사람용 `arch` 출력 옆에 `arch --json`을 추가한다. 텍스트 렌더러는 그대로 두고 JSON만 additive mode로 넣었다. 이 분리가 중요하다. 운영자는 읽기 쉬운 카탈로그가 계속 필요하고, recipes 빌더는 결정적인 스키마가 필요하기 때문이다.

### 2.3 스냅샷 생성

새 `recipes-registry` Make 타깃은 다음을 수행한다.

- release CLI를 빌드하고
- `mlxcel --version`에서 런타임 버전을 추출하고
- `recipes/registry/` 아래에 버전드 JSON 스냅샷을 쓰고
- `recipes/registry/CURRENT`를 갱신한다

이후 fix 커밋은 direct redirection을 temp-file-and-rename 흐름으로 바꾼다. 이 보강은 정확하다. 스냅샷은 이제 추적되는 제품 아티팩트이므로, 부분 쓰기는 단순 편의 문제가 아니라 정확성 버그다.

### 2.4 머지에 반영된 리뷰/보안 보정

머지된 PR에는 초기 기능 커밋 외에 실질적인 수정이 두 개 포함돼 있다.

- `fix: align arch registry KV modes with MLA resolver`
  DeepSeek V3와 V3.2는 generic capability table만 보면 정량화 KV 모드를 허용하는 것처럼 보였지만, 실제 MLA runtime path는 그 요청을 다시 `fp16`으로 내린다. 이 수정은 잘못 광고된 모드를 제거하고, runtime이 `fp16`으로 되돌리는 모드를 registry가 내보내지 못하게 하는 회귀 테스트를 추가한다.
- `fix: preserve recipes registry on rebuild failure`
  원래 Make 타깃은 JSON을 추적 파일에 바로 redirect했고, 실패 후에도 `CURRENT` 갱신으로 이어질 수 있었다. 수정 후에는 `set -e`, 임시 파일, cleanup trap, 성공 후 rename을 사용해 반쯤 써진 스냅샷이 공개되지 않게 한다.

이 둘은 외형 수정이 아니다. 기계 판독 export에서 가장 중요한 두 가지, 즉 계약의 진실성과 아티팩트 무결성을 닫는다.

### 2.5 호환성/의존성

- 새로운 crate 의존성은 없다.
- 기존 사람용 `mlxcel arch` 동작은 유지된다.
- 커밋된 스냅샷은 런타임 breaking change가 아니라 저장소 데이터 추가다.
- registry는 family-level capability metadata이지, 모든 checkpoint/backend 조합이 검증됐다는 보증이 아니다. 문서 업데이트가 이 경계를 명시한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 손으로 관리하는 카탈로그 대신 모델 capability에서 export한다

선택된 방식은 각 `ModelType`을 capability helper에 통과시켜 결과를 직렬화한다. 이렇게 해야 exported registry가 실제 실행 모델에 붙어 있고, 결국 드리프트할 수밖에 없는 두 번째 수작업 registry를 피할 수 있다.

버린 대안은 별도의 recipes JSON을 수작업이나 문서 prose에서 유지하는 방식이다. 그렇게 하면 코드와 콘텐츠 두 곳을 항상 같이 수정해야 하고, 패밀리 추가가 늦게 반영된다.

### 3.2 registry ID를 안정적인 제품 식별자로 취급한다

`registry_id()`와 `model_type_keys()` 매핑은 raw enum 이름만이 아니라 안정적인 family ID와 detection key를 JSON에 싣는다. 이것이 적절한 경계다. 하위 페이지와 recipe builder는 내부 Rust 표기법이 아니라 export된 ID에 묶여야 한다.

대가도 있다. 새 패밀리나 alias 표기가 추가될 때마다 매핑 유지비가 생긴다. 하지만 그 비용은 하위 호환이 조용히 깨지는 위험보다 훨씬 작다.

### 3.3 exported KV mode를 실제 runtime resolution에 맞춘다

후속 KV-mode 수정은 사실상 하나의 규칙을 세운다. registry는 그 패밀리에서 runtime이 실제로 수행할 수 있는 모드만 광고해야 한다. 이는 올바른 경계다. 단순 capability table의 낙관론을 반영하는 registry보다, 실제 runtime 행동에 닿아 있는 registry가 훨씬 유용하다.

### 3.4 생성 아티팩트는 원자적으로 쓴다

버전 관리되는 JSON은 제품 표면 일부다. temp file과 atomic rename을 쓰는 것은 그 사실을 인정하는 설계다. Make recipe는 조금 복잡해지지만, 실패한 rebuild 뒤에 깨진 스냅샷이 남는 훨씬 큰 위험을 제거한다.

---

## 4. 구현 상세

### 4.1 새 소스 모듈

`src/models/mod.rs`는 이제 `registry`를 export하고, `src/models/registry.rs`가 registry 스키마와 capability 유도 로직을 중앙화한다.

핵심 구현 seam:

- family-level capability 정의를 한곳에 모으고
- `registry_id()`가 공개 JSON 식별자를 매핑하고
- modality/runtime/backend/drafter/KV helper가 전체 모델 taxonomy를 정규화하고
- 테스트가 기존 dispatch contract와의 정합성을 검증한다

### 4.2 테스트

`src/main_tests.rs`와 `src/models/registry.rs` 테스트는 다음을 덮는다.

- `arch --json`의 CLI JSON 출력
- unique registry ID와 model type당 하나의 family coverage
- tensor-parallel / pipeline-parallel allowlist와 실행 contract의 정합성
- `qwen3`, `whisper`, rerank-only classifier 같은 대표 패밀리의 capability
- runtime resolution이 `fp16`으로 강등하는 KV mode를 registry가 광고하지 않는지
- MLA latent-cache 계열이 exported contract에서 `fp16` 전용으로 남는지

마지막 종류의 테스트가 특히 중요하다. "JSON에는 지원이라고 써 있는데 runtime은 조용히 강등"되는 회귀를 막아주기 때문이다.

### 4.3 문서와 커밋된 아티팩트

- `README.md`는 `mlxcel arch`와 `mlxcel arch --json`의 역할을 구분한다.
- `docs/supported-models.md`는 registry가 무엇을 담고 무엇은 보증하지 않는지 설명한다.
- `recipes/registry/0.6.0.json`과 `recipes/registry/CURRENT`가 첫 recipes-facing committed artifact가 된다.

---

## 5. 검증

PR 본문에는 다음 검증이 기록돼 있다.

| 검사 | 결과 |
|---|---|
| `cargo fmt --check` | 통과 |
| `cargo test --lib registry --no-default-features` | 통과 |
| `cargo test --bin mlxcel arch --no-default-features` | 통과 |
| `cargo check --lib --tests --no-default-features` | 통과 |
| `cargo clippy --lib --tests --no-default-features -- -D warnings` | 통과 |
| `make -n recipes-registry` | 통과 |
| `make recipes-registry` | 통과 |
| `diff -u /tmp/mlxcel-1606-arch.json recipes/registry/0.6.0.json` | 통과 |
| JSON parse/assert 스크립트 | 통과 |
| `arch --json` 실패 경로 시뮬레이션 | 추적 중인 registry artifact가 바뀌지 않음 |

이 검증은 실제 위험 표면과 맞아 있다. 단순 직렬화 단위 테스트에서 멈추지 않고, exported schema와 rebuild mechanics를 함께 확인한다.

---

## 6. 학습 포인트

**생성 카탈로그는 다른 도구가 소비하는 순간 공개 API가 된다.** recipes 워크플로가 파일에 의존하기 시작하면, 필드 이름, ID, capability 의미론은 CLI flag나 HTTP response처럼 다뤄야 한다.

**capability export는 추상적 가능성보다 실제 runtime 행동을 따라야 한다.** DeepSeek V3/V3.2 보정이 정확한 예다. runtime이 실제로 강등한다면 registry도 그렇게 말해야 한다.

**큰 생성 산출물에는 성공 경로뿐 아니라 실패 의미론이 필요하다.** Makefile의 atomic rewrite는 실패한 rebuild가 손상된 상태를 정상처럼 남기는 일을 막는다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 수 | 9 |
| 추가 줄 | 5977 |
| 삭제 줄 | 12 |
| 커밋 수 | 3 |

### 관련 커밋

- `3019b63` feat: add arch JSON recipes registry
- `5fbf2b3` fix: align arch registry KV modes with MLA resolver
- `d7acb2f` fix: preserve recipes registry on rebuild failure

### 주요 파일

| 파일 | 변경 |
|---|---|
| `src/models/registry.rs` | registry 스키마, capability 매핑, 계약 테스트 추가 |
| `src/main.rs` | `arch --json` 추가 |
| `src/main_tests.rs` | CLI JSON 동작 검증 추가 |
| `Makefile` | `recipes-registry` rebuild 로직 추가 및 보강 |
| `recipes/registry/0.6.0.json` | 첫 registry 스냅샷 |
| `recipes/registry/CURRENT` | 활성 스냅샷 버전 추적 |
| `README.md` | 기계 판독 모드 문서화 |
| `docs/supported-models.md` | registry 범위와 한계 설명 |

---

## 8. 후속 조치

- 새 패밀리 추가는 registry 표면 변경으로 취급하고, 같은 패치에서 `registry_id()`와 capability 테스트까지 확장해야 한다.
- 외부 도구가 exact field에 의존하기 시작하면, 별도 recipes-facing schema 문서를 두는 편이 낫다.
- 이후 버전이 additive evolution을 필요로 하면, 여러 committed snapshot 파일 간 호환성을 보는 경량 schema check를 추가하는 것을 검토할 수 있다.
