# 기술 보고서: PR #1605 - feat: add JSON inspect output

**작성일**: 2026-09-03
**작성자**: mlxcel maintainers
**리뷰어**: implementation review cycle
**상태**: 완료 (PR 본문에 CLI, 라이브러리, clippy, 로컬 실체크포인트 스모크가 기록돼 있다. 머지와 더 넓은 통합 게이트는 일반 PR 경로에서 진행된다)
**언어**: Rust, Markdown
**위험도**: Medium (`mlxcel inspect`에 새로운 기계 판독 계약이 추가되고, 캐시 미스 시 resolver 동작이 텍스트 모드와 JSON 모드에서 달라진다)

---

## 요약

레시피 빌더와 스케줄러는 기존 메모리 추정기를 안정적인 바이트 필드로 필요로 했지만, `mlxcel inspect`는 사람이 읽는 배너만 제공했다. 소비자는 전부 문장을 파싱해 단위를 추론해야 했고, 운영자를 위한 CLI 표현 형식과 강하게 결합될 수밖에 없었다.

PR #1605는 `mlxcel inspect --json`을 구조화된 계약으로 추가한다. 새 경로는 배너와 같은 추정기 상태를 그대로 재사용하고, 정확한 바이트 총량과 실제 입력값을 담은 JSON 객체 하나를 출력하며, `family`는 원시 `config.json` 이름이 아니라 공개 레지스트리와 맞물리는 classifier 매핑에서 유도한다. 또한 모델 해석은 조용한 offline 경로로 보내서 stdout이 항상 기계 판독 가능한 상태를 유지한다.

---

## 1. 문제 정의

### 1.1 배경

저장소에는 이미 `mlxcel inspect`, `generate --estimate-memory`, `serve --estimate-memory`가 함께 쓰는 통합 메모리 추정기가 있었다. 사람에게는 충분했지만 자동화에는 충분하지 않았다. 외부 레시피 도구는 모델 가중치, KV 캐시 총량, activation reserve, headroom, budget 같은 바이트 단위 필드를 필요로 했고, CLI는 서술형 텍스트 보고서만 내보냈다.

### 1.2 기존 문제점

- 도구가 안정적인 스키마 대신 사람이 읽는 문장을 파싱해야 했다.
- `config.json`의 원시 `model_type` 문자열은 공개 레지스트리 식별자와 항상 일치하지 않아, inspect 결과를 아키텍처 카탈로그와 안정적으로 조인할 수 없었다.
- 일반 모델 resolver는 bare-name 확장과 다운로드 안내를 stdout에 찍기 때문에, "JSON 객체 하나"를 약속하는 명령과 양립할 수 없었다.

### 1.3 방치 시 위험

구조화된 표면이 없으면 모든 하위 통합이 취약한 파서 로직을 중복 구현해야 하고, 표현만 바뀌는 CLI 변경에도 깨지게 된다. 더 큰 문제는 반쯤만 기계 판독 가능한 명령은 없는 것보다 더 나쁘다는 점이다. stdout에 데이터와 resolver 잡음이 함께 섞이면 자동화는 성공과 노이즈를 안전하게 구분할 수 없다.

---

## 2. 기술적 검토

### 2.1 새 계약

`src/execution/memory_estimate.rs`는 이제 `InspectReport`, `InspectReportInputs`, `InspectKvBytesPerToken`을 정의하고, `InspectReport::from_estimate()`가 텍스트 배너가 이미 출력하던 `MemoryEstimate` 값을 그대로 복사한다. 스키마에는 다음이 들어간다.

- 버전과 해석된 모델 경로
- 원시 `model_type`과 best-effort `family`
- 실제 입력값: `max_tokens`, `batch`, `kv_cache_mode`, `quant`
- 바이트 필드: `weights_bytes`, `kv_bytes_total`, `activation_bytes`, `headroom_bytes`, `budget_bytes`, `total_bytes`
- 계산 불가 시 `null`로 직렬화되는 선택 필드: 토큰당 KV 비율, paged per-slot overhead, family/model type

이 경계가 맞다. 보고서는 렌더된 텍스트를 다시 파싱하는 대신 추정기 상태에서 직접 만들어지므로, 텍스트와 JSON 형식이 조용히 어긋날 수 없다.

### 2.2 Resolver 동작

`src/commands/inspect.rs`는 JSON 모드에서 `resolve_model_source_quietly_with_options()`를 사용하고 `offline: args.json`을 설정한다. 의도는 두 가지다.

- 캐시 히트는 조용하게 지나가서 stdout이 정확히 JSON 객체 하나만 갖게 한다.
- 캐시 미스는 자동 다운로드 대신 stderr 실패로 끝나서, 다운로더 진행 메시지가 출력물을 오염시키지 않는다.

사람이 읽는 배너 경로는 이전 reuse-or-download 동작을 유지하므로, 이 PR은 대화형 운영자 경험을 바꾸지 않고 CLI 표면을 확장한다.

### 2.3 레지스트리 정렬

초기 JSON 경로는 raw `model_type`를 slugify하려 했지만, 그러면 `qwen3_5` 대 `qwen3-5`처럼 공개 레지스트리 id와 어긋날 수 있었다. 수정 커밋은 `family`를 `inspect_family_slug()`로 옮겨 `get_model_type()`과 아키텍처 카탈로그와 같은 registry-id 매핑을 통해 값을 유도한다. 원시 `model_type`는 별도 필드로 그대로 남긴다.

### 2.4 호환성/의존성 관점

- 새로운 의존성은 없다.
- 기존 `mlxcel inspect` 배너 출력은 유지된다.
- CLI 수준에서는 추가적 변경이다. `--json`이 새로 생겼을 뿐 기존 호출의 의미는 바뀌지 않는다.
- JSON 모드의 캐시 미스 offline 실패는 부수 효과가 아니라 계약 일부로 다뤄야 한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 배너를 파싱하지 않고 추정기 상태에서 JSON을 만든다

추정기는 이미 총량과 fit 판정을 위한 권위 있는 상태를 갖고 있었다. 그 상태를 재사용하면 이중 로직을 피하고, 기계 판독 경로와 사람이 읽는 경로가 같은 sizing 질문에 답하도록 강제할 수 있다.

버린 대안은 기존 배너를 다시 파싱하거나 재포맷하는 방식이다. 그 방식은 두 개의 진실 원천을 남기고, 설명 문구만 바뀌는 텍스트 수정도 자동화에 위험하게 만든다.

### 3.2 stdout을 깨끗하게 유지하기 위해 JSON 모드를 quiet + offline으로 둔다

기계가 읽는 CLI 출력은 stdout에 payload만 있어야 믿을 수 있다. 이 PR은 resolver 메시지를 꾸밈 문제가 아니라 `--json`의 정확성 문제로 취급한다. 단순히 quiet resolver만 도입해서는 부족했고, 캐시 미스가 downloader를 호출할 수 있으므로 offline 강제가 필요했다.

대가도 있다. `mlxcel inspect --json -m <repo-id>`는 캐시 미스에서 더 이상 자동 다운로드하지 않는다. 그러나 자동화 입장에서는 받아들일 만한 제약이다. 이제 스크립트는 성공 시의 모양과 실패 채널을 모두 결정적으로 가정할 수 있다.

### 3.3 `family`와 raw `model_type`를 분리한다

두 필드는 역할이 다르므로 둘 다 노출하는 편이 맞다.

- `model_type`는 체크포인트의 원시 메타데이터를 보존한다.
- `family`는 레시피 도구가 실제로 필요로 하는 레지스트리 조인 키를 제공한다.

하나로 합치면 원시 메타데이터를 잃거나, 반대로 공개 도구에 내부 이름 흔들림을 그대로 새기게 된다.

### 3.4 없는 값은 가짜 기본값 대신 `null`로 둔다

기하 정보가 없어 계산할 수 없는 필드를 `0`이나 임의 문자열로 채우지 않고 `null`로 직렬화한다. 이렇게 해야 "계산 불가"가 드러나고, 소비자가 TurboQuant 크기처럼 아직 지원하지 않는 경로를 실제 0 비용으로 오해하지 않는다.

---

## 4. 구현 상세

### 4.1 CLI 표면

`src/main.rs`는 `InspectArgs`에 `json: bool`을 추가해 `mlxcel inspect --json`이 같은 추정치를 JSON 객체 하나로 출력한다고 문서화한다.

### 4.2 명령 흐름

`src/commands/inspect.rs`는 이제:

- `--json`일 때 quiet resolver로 모델 경로를 해석하고
- 실제 `generate`와 `serve`가 만들 KV 캐시 모드를 그대로 해석한 뒤
- `serde_json::to_string_pretty(&report)`를 출력하고 조기 반환한다

텍스트 배너 경로는 그대로 유지되며, fitting 실패 시의 안내 문구도 유지된다.

### 4.3 추정기 확장

`src/execution/memory_estimate.rs`는 JSON 구조체와 함께 다음 도우미를 추가한다.

- `config.json`에서 raw model type을 추출하는 함수
- 레지스트리 id와 정렬된 best-effort family 분류 함수
- 기본 paged decode 계열에 대한 per-slot overhead 보고 함수

단위 테스트는 계약 복사, 안정적인 키 순서, null 직렬화, raw와 classified family 값의 차이, `qwen3_5`와 `gemma3_text` 같은 레지스트리 경계 사례를 검증한다.

### 4.4 문서

- `README.md`는 `mlxcel inspect --json` 예시와 핵심 바이트 필드를 추가한다.
- `docs/environment-variables.md`는 `MLXCEL_MEMORY_LIMIT`가 inspect와 estimate-memory preflight에도 같은 파서로 반영된다는 점을 명확히 한다.

---

## 5. 검증

PR 본문에 기록된 검증은 다음과 같다.

| 검사 | 결과 |
|---|---|
| `cargo fmt --check` | 통과 |
| `cargo check --lib --tests` | 통과 |
| `cargo test --lib -- memory_estimate` | 통과 |
| `cargo clippy --lib --tests -- -D warnings` | 통과 |
| `/home/inureyes/models/mlx/qwen3-4b-4bit` 실체크포인트 스모크 | 텍스트 배너 유지, JSON 모드는 빈 stderr와 함께 stdout JSON 객체 1개 출력 |
| `definitely-missing-model-for-pr1605` 캐시 미스 스모크 | stdout 비어 있음, 실패는 stderr로만 노출 |

이 검증은 변경 범위와 잘 맞는다. 추정기 단위 경계, CLI 통합 경로, 그리고 캐시 히트/미스 양쪽에서의 structured-output 청결성을 모두 확인한다.

---

## 6. 학습 포인트

**구조화된 CLI 출력은 별도의 제품 표면이다.** 명령이 기계 판독을 약속하는 순간 resolver 메시지나 다운로드 진행 출력은 더 이상 무해한 UX가 아니라 계약 위반이 된다.

**원시 메타데이터와 공개 식별자는 같은 것이 아니다.** `config.json`은 출처 정보로 유용하지만, 자동화는 종종 정제된 카탈로그와 맞물리는 안정적인 조인 키를 필요로 한다. 두 필드를 함께 노출하면 책임이 분리된다.

**가짜 확정보다 nullable이 낫다.** 계산할 수 없는 필드를 `null`로 두면 추정기의 실제 상태가 드러나고, 아직 지원되지 않은 경로에 대해 하위 소비자가 잘못된 가정을 굳히는 일을 막을 수 있다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 수 | 7 |
| 추가 줄 | 561 |
| 삭제 줄 | 20 |
| 커밋 수 | 3 |

### 관련 커밋

- `f8d670e` feat: add JSON inspect output
- `c142dbd` fix: align inspect family with registry ids
- `85eea20` fix: keep inspect JSON stdout machine-readable

### 주요 파일

| 파일 | 변경 |
|---|---|
| `src/main.rs` | `inspect`에 `--json` 추가 |
| `src/commands/inspect.rs` | JSON 모드를 quiet offline 해석과 report 직렬화로 연결 |
| `src/execution/memory_estimate.rs` | JSON 계약과 계약 중심 테스트 추가 |
| `src/downloader/resolver.rs` | structured-output 호출자를 위한 quiet model resolution 노출 |
| `README.md` | 새 JSON 형식 문서화 |
| `docs/environment-variables.md` | memory-limit와 inspect/preflight 연계를 명시 |

---

## 8. 후속 조치

- 이슈 `#1508`이 `main`에 들어오면, 로컬 `inspect_family_slug()` fallback을 정식 `mlxcel arch --json` registry id로 교체한다.
- 외부 도구가 이 JSON 형태에 의존하기 시작하면, 이후 확장이 additive로 유지되도록 전용 문서나 호환성 노트를 두는 편이 좋다.
- 다른 structured CLI 표면도 같은 "stdout은 조용하게, 실패는 stderr" 규칙을 따를지 검토하면 자동화 의미론을 명령 간에 더 일관되게 만들 수 있다.
