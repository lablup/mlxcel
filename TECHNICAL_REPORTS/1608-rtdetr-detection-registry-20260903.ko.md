# 기술 보고서: PR #1608 - fix: include RT-DETRv2 in arch JSON registry

**날짜**: 2026-09-03
**작성자**: mlxcel 유지보수자
**검토**: 구현, 보안, 최종화 검토 사이클
**상태**: 완료 (집중 Rust 검사와 hosted CI 통과)
**언어**: Rust, JSON, Markdown
**위험 수준**: 중간 (registry는 downstream recipes 계약이며, 부정확한 runtime 광고는 실행 불가능한 명령을 생성할 수 있음)

---

## 요약

PR #1608은 PR #1606 이후 남아 있던 coverage 누락을 교정합니다. 최초 `mlxcel arch --json` registry는 `ALL_MODEL_TYPES`만으로 만들어졌습니다. 이 목록은 text, VLM, embedding, reranking, speech, audio 모델 로더를 나타내지만, RT-DETRv2는 별도 `mlxcel detect` 명령으로 지원되기 때문에 실제 runtime이 registry에서 조용히 빠졌습니다.

이번 교정은 standalone architecture family를 위한 결정적 확장 지점을 추가하고, `rt_detr_v2`가 실제로 제공하는 capability만 등록합니다. 즉 `detect` runtime, `image` 입력, `boxes` 출력, `detection` category이며, 생성·서빙·분산 실행·speculative decoding은 광고하지 않습니다. 보안 검토 후속 커밋은 standalone 항목이 loader-backed family ID 또는 model-type key와 충돌하지 못하도록 invariant도 추가했습니다.

---

## 1. 문제 정의

### 1.1 배경

recipes catalog는 versioned architecture registry를 사용해 어떤 mode와 command 형식을 제공할 수 있는지 판단합니다. PR #1606은 이 정보를 기계 판독 가능하게 만들었지만, family 수집 과정은 모든 runtime이 주 모델 로더 enum에 들어 있다는 가정을 사용했습니다.

RT-DETRv2는 의도적으로 이 가정의 바깥에 있습니다. 객체 검출은 image를 입력받아 bounding box를 출력하므로 token 생성과 다른 predictor와 CLI handler를 사용합니다. 일반 생성 모델처럼 취급하면 잘못된 capability가 생기고, 제외하면 registry가 불완전해집니다.

### 1.2 사용자 영향

- Recipe builder가 `detect` runtime을 발견할 수 없었습니다.
- `detection` category와 `boxes` 출력을 제공하는 registry 항목이 없었습니다.
- 유효한 RT-DETRv2 recipe를 exported capability catalog로 검증할 수 없었습니다.
- registry 열거만을 위해 RT-DETRv2를 `ModelType`에 넣으면 거짓 `generate` 또는 `serve` 지원이 생길 위험이 있었습니다.

---

## 2. 변경 요약

### 2.1 Standalone architecture family

`src/models/registry.rs`에 text/VLM 모델 로더를 통과하지 않는 command runtime용 작은 standalone-family descriptor를 추가했습니다. `build_architecture_registry()`는 기존 `ALL_MODEL_TYPES` 순서를 유지하고, 선언된 고정 순서대로 standalone 항목을 뒤에 추가합니다.

첫 standalone 항목은 `rt_detr_v2`입니다.

| 필드 | 값 |
|---|---|
| `id` | `rt_detr_v2` |
| `model_types` | `rt_detr_v2` |
| `runtimes` | `detect` |
| `modalities_in` | `image` |
| `output` | `boxes` |
| `category` | `detection` |
| TP / PP | 비활성 |
| drafters | 없음 |

실제 `mlxcel detect` 경계에 맞게 `generate`와 `serve`는 포함하지 않습니다.

### 2.2 충돌 invariant

보안 검토에서는 확장 지점의 유지보수 위험을 발견했습니다. 이후 standalone 항목이 기존 family ID 또는 detection key를 재사용하면 downstream map에서 loader-backed family를 조용히 덮을 수 있습니다. 후속 테스트는 exported identity set을 구성해 다음을 거부합니다.

- 중복 standalone ID,
- `ALL_MODEL_TYPES` registry ID와 충돌하는 standalone ID,
- 중복 standalone model-type key,
- loader-backed family key와 충돌하는 standalone key.

### 2.3 Snapshot과 문서

Release CLI로 `recipes/registry/0.6.0.json`을 재생성해 schema 변경 없이 detector 항목을 추가했습니다. `README.md`와 `docs/supported-models.md`는 이제 registry가 loader-backed family와 standalone command runtime을 함께 포함한다고 설명합니다.

---

## 3. 기술적 선택과 그 이유

### 3.1 Detector runtime을 `ModelType` 밖에 유지

`ModelType`은 text/VLM loading과 capability dispatch를 구동합니다. Registry 열거만을 위해 RT-DETRv2를 추가하면 image-to-box predictor가 token-generation 가정에 결합되고, 잘못된 runtime 광고 가능성이 커집니다.

Standalone descriptor는 runtime 소유권을 사실대로 유지하면서 recipes consumer에는 하나의 통합 JSON 문서를 제공합니다.

### 3.2 결정적 순서 보존

기존 loader-backed 항목은 `ALL_MODEL_TYPES` 순서를 그대로 유지합니다. Standalone family는 map이 아니라 static slice에서 추가되므로 반복 빌드 결과가 byte-stable하고 committed snapshot을 쉽게 검토할 수 있습니다.

### 3.3 Loader coverage를 재정의하지 않고 count 확장

Registry family 수는 이제 loader-backed family 수와 standalone runtime family 수의 합입니다. 테스트가 이 공식을 명시적으로 표현하므로, `ALL_MODEL_TYPES` exhaustiveness를 유지하면서 enum 밖의 지원 command도 인정합니다.

### 3.4 모호한 공개 identity를 테스트에서 차단

Downstream recipe index는 흔히 family ID와 model-type key를 키로 사용합니다. 충돌을 테스트 시점에 거부하면 last-writer-wins 형태의 조용한 catalog 손상을 즉시 개발 실패로 바꿀 수 있습니다.

---

## 4. 검증

| 검사 | 결과 |
|---|---|
| `cargo fmt --check` | 통과 |
| `cargo test --lib registry --no-default-features` | 통과 |
| `cargo test --bin mlxcel arch --no-default-features` | 통과 |
| `cargo check --lib --tests --no-default-features` | 통과 |
| `cargo clippy --lib --tests --no-default-features -- -D warnings` | 통과 |
| `make recipes-registry` | 통과 |
| Release JSON과 committed snapshot 비교 | byte-identical |
| RT-DETRv2 capability assertion | 통과 |
| Standalone ID/key collision assertion | 통과 |
| Hosted 필수 검사 | 통과 |

최종화 시 release registry와 committed snapshot의 SHA-256은 모두 `3360ff0554365c702e9eb501d85c0ec5ae8d4dd7aadddcdc4059be81efcdfddf`였습니다.

---

## 5. 변경 통계

| 지표 | 값 |
|---|---|
| 변경 파일 | 5 |
| 추가 라인 | 204 |
| 삭제 라인 | 35 |
| 커밋 | 2 |

관련 커밋:

- `0ab5509` fix: include RT-DETRv2 in arch JSON registry
- `9cd2806` test: guard standalone registry extension collisions

주요 파일:

- `src/models/registry.rs`: standalone descriptor, registry 조립, capability 및 collision 테스트
- `src/main_tests.rs`: CLI 수준 family count와 detector assertion
- `recipes/registry/0.6.0.json`: 재생성된 공개 snapshot
- `README.md`, `docs/supported-models.md`: registry 범위 명확화

---

## 6. 학습 포인트

**Runtime catalog는 loader enum뿐 아니라 command boundary를 열거해야 합니다.** 단일 enum은 실제로 모든 실행 경로를 소유할 때만 완전한 source of truth입니다.

**과장된 capability보다 사실에 맞는 제한이 낫습니다.** Detector는 registry에 있어야 하지만 구현하지 않는 생성·서빙 capability를 물려받아서는 안 됩니다.

**확장 지점에는 처음부터 identity invariant가 필요합니다.** Static data도 충돌할 수 있고 downstream JSON consumer는 이를 조용한 덮어쓰기로 바꾸기 쉽습니다.

---

## 7. 후속 작업

- 향후 standalone runtime도 같은 static descriptor와 collision 테스트를 통해 추가합니다.
- Recipe builder는 additive family를 허용하되 기존 ID와 field 의미는 안정적인 계약으로 다룹니다.
- RT-DETRv2에 backend별 qualification 자료가 생기면 backend support를 다시 검증합니다.

