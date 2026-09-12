# 기술 리포트: PR #1868 — 모델 카탈로그 투영

**작성일**: 2026-09-12
**상태**: 열린 PR; 최종 집중 CPU 검증과 기존 루트 전체 workspace·실모델 게이트 완료, GB10 CI 사용 불가
**언어**: Rust, JSON/OpenAPI, TypeScript, Markdown
**위험도**: 중간
**구현 스냅샷**: 최종 캐시 소유권·symlink 증거 강화 업데이트 이후 PR #1868 브랜치 head

## 요약

[PR #1868](https://github.com/lablup/mlxcel/pull/1868)은 에픽 #1834의 이슈 #1840을 위한 메타데이터 전용 카탈로그 기반을 구현합니다. 기존 라우터 인벤토리와 provider 상태를 타입이 있는 목록·상세 응답 및 coordinator 소유의 새로고침 작업으로 투영하며 병렬 모델 레지스트리를 만들지 않습니다. 프로덕션 시작 시 마운트는 후속 #1838 범위이며 이 리포트는 사용자용 WebUI 전체의 완성을 주장하지 않습니다.

## 1. 문제 정의

체크포인트 다운로드 여부, 인식 가능한 아키텍처, 지원되는 백엔드, 준비된 provider는 서로 다른 사실입니다. 디렉터리 이름이나 공급업체 제목으로 카탈로그를 만들면 이들을 혼동하고 현재 런타임이 실행할 수 없는 채팅·이미지 작업을 제시할 수 있습니다. 목록 조회에 제한 없는 로더 감지를 재사용하면 가중치·헤더를 읽거나 요청 executor에서 불필요한 작업을 수행할 위험도 있습니다.

기존 캐시·모델 디렉터리·프리셋의 권한, 안정적 식별자, 수명주기 coordinator를 유지하면서 불완전하거나 미지원인 체크포인트도 사실대로 표시해야 합니다. 단일 모델 모드는 새 provider를 등록하지 않고 기존 provider를 설명해야 합니다.

## 2. 기술적 선택과 그 이유

### 하나의 감지 판단과 서로 다른 probe

`ModelDetectionProbes`는 기존 감지 dispatch와 파일시스템 증거 획득을 분리합니다. 실제 로드는 일반 probe 구현을 유지합니다. 카탈로그는 제한된 config·sidecar·index probe를 제공하고 SafeTensors 헤더가 필요한 분류는 거절합니다. 새 수기 family 테이블 대신 레지스트리에서 task와 백엔드 지원을 도출합니다.

이 선택은 의도적으로 불확실성을 유지합니다. 충분한 메타데이터나 provider 증거가 없다면 체크포인트를 조회할 수 있어도 변형을 확정하지 않습니다. Vision/MTP 증거가 없을 때 “텍스트 모델”로 추정하면 편리하지만 사실과 다른 답이 됩니다.

### 메타데이터는 캐시하고 수명주기는 현재 상태로 유지

카탈로그는 blocking worker에서 `RouterPool::catalog_snapshot()`을 투영합니다. 메타데이터 캐시는 각 `RouterServerState` 또는 단일 모델 catalog context가 소유하고 안정 ID와 라우터 catalog epoch, 모델 generation을 키로 사용하며 현재 revision, lifecycle, provider가 확인한 capability, 제거 가능 여부를 다시 적용합니다. 일반 GET 캐시 적중은 content fingerprint를 다시 계산하거나 무거운 메타데이터 획득을 반복하지 않으며, 최초 캐시 채우기와 명시적 새로고침은 제한된 검사를 수행합니다. Fingerprint는 가중치 내용의 체크섬이 아닙니다. 한 라우터 풀이 다른 풀의 catalog metadata를 지우거나 축출하거나 재사용할 수 없습니다.

기본 페이지 크기는 50, 최대 200이며 인벤토리는 1,000개로 제한합니다. 결과는 불투명 ID 순으로 정렬합니다. 안정성은 인벤토리가 변하지 않을 때의 보장이며 동시 변경 중 요청 간 트랜잭션을 제공하지 않습니다.

### 실행 소유권은 작업 재사용보다 강한 조건

새로고침은 coordinator 작업이 활성 상태인 동안 서버 인스턴스별로 하나의 실행 소유자를 둡니다. 접수·재사용한 작업을 반환하는 것만으로는 여러 백그라운드 재탐색을 막을 수 없습니다. 소유자만 재탐색을 시작하고 종료 시 정리하여 이후 새로고침을 허용합니다. Signature 비교는 전후 항목 개수가 같아도 감지된 변경을 계산합니다.

### API 제공과 통합 준비 상태를 분리

인증된 라우터 생성자는 테스트와 향후 안전한 시작 경로를 위해 카탈로그 handler를 노출하지만 일반 생성자는 UI API를 마운트하지 않습니다. 단일 모델 accessor는 기존 `AppState`의 provider와 실제 추론 ID를 읽고 읽기 전용 제거 안내를 반환하며 두 번째 provider를 만들지 않습니다. 브라우저 보안과 프로덕션 CLI 통합은 #1837/#1838, 실제 삭제 실행은 #1841의 책임입니다.

## 3. 리뷰 보완 사항

다음 보완 후 독립적인 정확성·보안 리뷰에서 최종 구현 스냅샷을 승인했습니다.

- 제한된 메타데이터 읽기와 공유 감지 probe로 제한 없는 가중치 헤더 검사로의 fallback을 막고 근거 없는 추정 대신 unknown 사유를 제공합니다.
- Symlink sidecar, config 파일, SafeTensors index 파일, shard 증거, 라우터 cache 항목, models-dir 항목, preset 경로, 중첩 pooling 부모를 증거에서 제외합니다. 마지막 수정은 leaf 파일만 확인하지 않고 `1_Pooling` symlink 경유를 거절합니다.
- 메타데이터 캐시는 라우터 서버 또는 단일 모델 catalog context별로 소유되며, 캐시한 항목의 오래된 readiness를 재사용하지 않고 현재 풀의 lifecycle·provider 사실을 적용하고, 매 GET fingerprint 순회가 아니라 라우터 catalog epoch를 따릅니다.
- 원본 `model_type`은 스키마 한도 안에서 정확히 보존하고 `declared_architectures`는 제한된 nullable 원본 배열로 노출하며, 문자열이 아니거나 너무 긴 값은 잘라낸 가짜 정확성이 아니라 사유가 있는 null이 됩니다.
- 새로고침 singleflight는 작업 접수와 실행 소유권을 분리하고 signature를 통해 동일 개수의 변경도 보고합니다. 동일 크기 config/index 편집, 레거시 reload, 캐시 download 회귀 테스트를 추가했습니다.
- Producer 테스트는 고정된 스키마 검증 fixture와 직렬화된 카탈로그 전체 구조를 비교하며 임시 경로에서 파생한 ID·fingerprint와 디스크 바이트만 정규화합니다.

Config·sidecar는 256 KiB, index JSON은 512 KiB로 읽기를 제한합니다. 디스크 순회는 방문·대기 항목 4,096개와 깊이 8을 제한으로 사용하고 측정할 수 없는 값은 사유가 있는 null로 반환합니다. 측정한 지연시간이나 메모리 감소를 주장하지 않습니다.

## 4. 검증 기록

아래는 구현 및 독립 리뷰 단계에서 최종 집중 변경에 대해 전달한 결과입니다. 문서 작성 단계에서는 빌드·테스트·모델 실행을 시작하지 않았습니다.

| 게이트 | 리포트 작성 시점 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate catalog -- --nocapture` | 통과: library catalog/router 테스트 32개와 CLI help 테스트 1개 |
| `cargo test --profile test-fast --features metal,accelerate router_models_discovery_tests -- --nocapture` | 통과: router source discovery symlink 회귀 2개 |
| `cargo check --no-default-features --features metal,accelerate --lib --tests` | 기존 feature-off warning과 함께 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 통과 |
| `cargo fmt --check`, `git diff --check`, `python3 scripts/insert_apache_header.py --check` | 통과 |
| `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` | fixture 40개 통과 |
| `python3 scripts/ci/check_cross_repo_refs.py`, `python3 scripts/ci/check_kernel_dtype_keys.py`, `/tmp/mlxcel-webui-contract/bin/python scripts/ci/check_crate_versions.py` | 통과; cross-repo 검사는 새 bare ref들이 같은 저장소 mlxcel 이슈·PR인지 수동 확인을 요청함 |
| 루트의 전체 workspace·로컬 CI와 실제 Llama + Granite 회귀 검증 | 통과: 11187/0/361과 clippy, Llama 572자 smoke, Granite affirmative smoke, SIGINT worker-exit 1/1 |
| GB10 필수 CI | 러너가 down 상태임을 확인했고 사용자가 해당 불가 게이트 생략을 명시적으로 승인 |
| CUDA 실행, 프로덕션 `--webui`, 브라우저 수용 검증 | 이 변경의 집중 증거로 확립하지 않음 |

러너 예외는 사용 불가능한 GB10 CI에만 적용합니다. 로컬 실패를 면제하거나 CUDA 컴파일·추론 성공을 뜻하지 않습니다. 직렬화된 루트 게이트는 전체 workspace·로컬 CI와 실제 dense·hybrid smoke로 공유 감지 변경 위험을 확인했습니다. 프로덕션 WebUI 시작과 브라우저 수용 검증은 후속 범위입니다.

## 5. 변경 요약

| 영역 | 변경 |
|---|---|
| 감지 판단 | 제한된 카탈로그 probe를 주입할 수 있는 공유 dispatch |
| 카탈로그 투영 | 타입이 있는 식별자·메타데이터·지원 사유·capability·제거 안내·필터·페이지 처리 |
| 라우터 어댑터 | 인증된 목록·상세·새로고침 accessor, singleflight 실행, 새로고침 오류 경로 숨김 |
| 기존 provider 통합 | 현재 풀 capability 스냅샷과 단일 모델 `AppState` accessor |
| 계약 | OpenAPI, 생성 TypeScript, 카탈로그·식별자 fixture의 동시 갱신 |
| 테스트 | 전체 producer 계약, 제한된 파일시스템 증거, 라우터별 캐시 격리, 캐시된 단일 모델 투영, 최신 lifecycle/provider 투영, HTTP 캐시 적중 metadata 획득 0회, 동일 크기 새로고침 편집, reload/download 무효화, 1,000개 HTTP 순회 |
| 문서 | 영어·한국어 통합 안내와 이 머지 전 리포트 |

구현 커밋은 `57c1f268`(투영과 공유 probe), `70809247`(필수 테스트 라이선스 헤더), `fafc5cf5`(pooling 부모 증거 강화와 전체 HTTP 새로고침 페이지 검증), `6befcd26`(원본 제한 메타데이터 복원과 epoch 기반 카탈로그 메타데이터 캐시)입니다. 최종 강화 업데이트는 캐시와 metadata 획득 counter를 라우터 또는 단일 모델 catalog context별로 한정하고, 카탈로그와 라우터 source discovery 전체에서 symlinked config/index/shard 증거를 거절하며, 캐시 항목에도 최신 provider/lifecycle 투영을 유지합니다.

## 6. 학습 포인트와 후속 조치

읽기 전용 관찰자에게 필요한 것은 별도 모델 분류 체계가 아니라 제한된 증거 획득 경계입니다. 또한 캐시 적중 시 비싼 메타데이터는 재사용해도 최신 lifecycle 사실은 필요하며 멱등 응답이 자동으로 단일 실행을 뜻하지 않습니다.

후속 통합은 null·사유 의미를 유지하고 제어에는 카탈로그 ID, 추론에는 inference ID를 사용하며 변경 작업의 권한 경계에서 다시 검사해야 합니다. 인증된 handler 테스트만으로 추정하지 말고 프로덕션 시작·보안을 검증해야 합니다.

[카탈로그 통합](../docs/webui/catalog.ko.md), [API 계약](../docs/webui/api.yaml), [아키텍처](../docs/webui/architecture.md)를 참고하십시오.
