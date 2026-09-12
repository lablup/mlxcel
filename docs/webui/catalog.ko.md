# 모델 카탈로그 통합

[English](catalog.md)

## 범위와 책임

이슈 #1840은 기존 `RouterPool`의 메타데이터 전용 카탈로그 투영과 목록·상세·새로고침 어댑터를 추가합니다. 별도 모델 레지스트리를 만들거나 새 provider를 시작하지 않습니다. `create_router_app_with_authenticated_ui`는 통합 테스트와 향후 안전한 시작 경로를 위해 필수 API 키 인증 뒤에 어댑터를 노출합니다. 일반 `create_router_app`에는 아직 마운트하지 않습니다. 프로덕션 `--webui` 시작과 브라우저 보안 통합은 각각 #1838과 #1837의 범위이며, 아래 API 경로가 프로덕션 플래그의 제공을 뜻하지는 않습니다.

탐색의 소유자는 여전히 라우터입니다. 관리형 캐시, 명시적인 `--models-dir`, 프리셋은 기존 충돌 우선순위(캐시 < 모델 디렉터리 < 프리셋), 별칭, 숨김 정책을 유지합니다. 저장소의 모델은 `--models-dir models/mlx`로 명시적으로 선택합니다. 카탈로그는 현재 작업 디렉터리에 따라 달라지는 기본값이나 파일시스템 선택기를 추가하지 않습니다. 목록 조회는 발견된 항목을 투영하며 체크포인트 다운로드, tokenizer 열기, provider 생성, 가중치 로드를 하지 않습니다.

## 투영 데이터 사용

아래 어댑터 경로는 향후 검증된 서버 API 접두사를 기준으로 합니다. 전체 DTO는 [API 스키마](api.yaml)와 [생성된 TypeScript 선언](generated/ui-api.d.ts)을 확인하십시오.

| 요청 | 동작 |
|---|---|
| `GET /ui-api/v1/catalog` | 불투명 카탈로그 ID 순으로 정렬한 필터링 목록 |
| `GET /ui-api/v1/catalog/{id}` | 표시 가능한 항목 하나 또는 구조화된 not-found 오류 |
| `POST /ui-api/v1/catalog/refresh` | 설정된 소스를 명시적으로 재탐색하는 수명주기 coordinator 작업 접수 |

목록은 `limit`(기본 50, 최대 200), 반환받은 `cursor`, `q`(최대 128바이트), `source`, `task`, `lifecycle`, `support`, `completeness`를 받습니다. 커서는 최대 512바이트이며 투영 대상은 1,000개 항목으로 제한됩니다. 페이지 순서는 인벤토리가 변하지 않을 때 결정적이며 동시 새로고침을 가로지르는 트랜잭션 스냅샷은 아닙니다. `server_instance_id`와 `snapshot_sequence`를 보관하고 상태 변경 시 [architecture.md](architecture.md)의 재스냅샷 규칙을 따릅니다.

카탈로그 작업과 선택에는 `identity.id`, 추론 요청에는 `identity.inference_id`를 사용합니다. 표시 이름은 어느 쪽의 식별자도 아닙니다. 콘텐츠 fingerprint는 메타데이터를 처음 투영하거나 명시적으로 새로고침할 때 관찰한 제한된 파일시스템 메타데이터를 나타내며, 가중치 내용의 암호학적 검증값도 아니고 매 polling마다 다시 계산하는 값도 아닙니다. revision과 lifecycle은 브라우저가 관리하는 별도 상태 머신이 아니라 풀에서 가져옵니다.

다음 사실을 하나의 “작동함” 배지로 합치지 마십시오.

- `complete`는 로컬 체크포인트 파일 구성을 나타내며 로드 성공이나 텐서 무결성 검증이 아닙니다.
- `metadata.support.architecturally_supported`는 공유 모델 감지와 아키텍처 레지스트리에서 도출하며 공급업체 제목의 일치 여부가 아닙니다.
- `runnable_on_backend`는 컴파일된 백엔드에 대한 레지스트리 지원 상태이며 실제 추론 측정 결과가 아닙니다.
- `tested_checkpoint`는 현재 명시적인 사유와 함께 false입니다. 카탈로그에는 체크포인트별 검증 증거 데이터베이스가 없습니다.
- `lifecycle.state`는 현재 provider 수명주기를 나타냅니다. 로드 전 capability가 있다고 `ready`인 것은 아닙니다.

`metadata.model_type`은 `config.json`의 원본 문자열을 제한 안에서 그대로 보존한 값이며 대소문자를 유지합니다. 값이 없거나 문자열이 아니거나 너무 길거나 읽을 수 없으면 잘라내지 않고 사유가 있는 `null`을 반환합니다. `metadata.declared_architectures`도 같은 무절단 규칙을 적용한 원본 `architectures` 배열입니다. `metadata.architecture`는 이 원본 필드의 단순 복사본이 아니라 공유 로더 감지 권한이 해석한 mlxcel 레지스트리 식별자입니다. 카탈로그는 그 권한에 제한된 probe를 제공합니다. 변형 구분에 가중치 헤더가 필요하고 제한된 sidecar 증거가 없으면 사유와 함께 unknown으로 남깁니다. 관련 Gemma 4, Inkling, Kimi K3 변형을 임의로 텍스트 모델이라고 추정하지 않는 것도 이 원칙에 포함됩니다.

Capability에는 적용 단계와 사용할 수 없는 사유가 있습니다. 이미지 입력은 준비된 기존 provider에서 확인하며 메타데이터만으로 이미지 전송을 활성화하지 않습니다. 비채팅 출력 task는 채팅과 구분합니다. 알 수 없는 파라미터 수와 메모리 추정값은 0이 아니라 사유가 있는 `null`입니다. 디스크 바이트는 파일 크기이며 프로세스 RSS나 allocator 메모리가 아닙니다.

## 파일시스템과 새로고침 경계

Config와 분류 sidecar는 256 KiB, SafeTensors index JSON은 512 KiB 읽기 제한을 적용합니다. 카탈로그 probe는 SafeTensors 헤더나 payload를 읽지 않습니다. 디스크 계산은 방문·대기 항목 최대 4,096개와 깊이 8을 적용하고 symlink를 건너뛰며 제한 안에서 완료할 수 없으면 사유와 함께 `null`을 반환합니다. 중첩된 `1_Pooling/config.json`은 부모 구성요소가 symlink가 아닌 실제 디렉터리일 때만 증거로 인정합니다.

메타데이터 투영은 blocking worker에서 실행합니다. 각 라우터 또는 단일 모델 WebUI 컨텍스트가 자체 제한 캐시를 소유하므로 한 풀의 새로고침이나 1,000개 항목 eviction이 다른 풀의 warm 투영을 무효화하지 않습니다. 캐시는 일반 polling의 캐시 적중에서 재귀 디스크 계산, config 파싱, 콘텐츠 fingerprint 계산을 포함한 제한된 메타데이터 획득을 반복하지 않도록 합니다. 다만 lifecycle, revision, provider가 확인한 capability, 제거 가능 여부는 현재 풀 상태로 갱신합니다. 최초 캐시 채우기와 명시적 새로고침은 제한된 파일시스템 검사를 수행합니다. 명시적인 새로고침은 소유 컨텍스트의 캐시만 비우고 기존 라우터 재탐색을 실행하며, 레거시 라우터 reload도 카탈로그 epoch를 전진시켜 캐시된 메타데이터가 영구히 stale로 남지 않게 합니다.

작업이 활성 상태인 동안 서버 인스턴스마다 하나의 새로고침만 실행권을 가집니다. 동시 요청은 별도 재탐색을 시작하지 않고 같은 작업을 재사용하며 완료 후의 요청은 새 작업을 시작할 수 있습니다. `changed_entries`는 항목 signature를 비교하여 개수가 같아도 추가·삭제·감지된 변경을 포함합니다. 클라이언트에 전달하는 새로고침 오류는 경로를 숨기고 진단 상세는 서버 로그에 남깁니다.

제거 가능 여부는 안내 정보이지 파일 삭제 권한이 아닙니다. 관리형 캐시만 제거 대상이 될 수 있으며 busy 항목은 사용할 수 없습니다. 실제 제거와 작업 경계의 검사는 #1841 범위입니다. 단일 모델 모드는 cache-aware `single_model_entry_from_state_with_cache(&CatalogProjectionCache, &AppState)` handoff로 기존 provider와 실제 추론 ID를 설명하고 별도 provider를 등록하지 않으며, 캐시된 정적 메타데이터 위에 최신 provider/lifecycle 상태를 투영하고 읽기 전용 제거 사유를 반환합니다. 프로덕션에서 이 accessor를 연결하는 작업은 #1838 범위입니다.

## 회귀 테스트 범위

집중 테스트는 스키마로 검증된 fixture와 실제 producer 전체 JSON의 비교, 원본 model_type·declared_architectures 제한, unknown 메타데이터, cache epoch 무효화, HTTP 캐시 적중의 메타데이터 획득 0회, 명시적 새로고침 후 동일 크기 config/index 편집 반영, 최신 lifecycle/provider 투영, 동시 1,000개 항목 카탈로그에서 라우터별 캐시 격리, 공유 감지, symlink 증거, 캐시된 기존 단일 provider 접근, HTTP 새로고침 전후 1,000개 항목의 전체 순회를 검증합니다. 공유 감지 변경 후의 추론 회귀 검사나 향후 프로덕션·브라우저 보안 수용 검증을 대신하지는 않습니다. 검증 기록과 환경 예외는 [PR #1868](https://github.com/lablup/mlxcel/pull/1868)을 확인하십시오.

최종 통합 게이트는 `808994e353fdab5563966e751ca2c71515d08595`에서 실행했습니다. 카탈로그 library 32개 + CLI 1개, 보안 26개, discovery 2개와 workspace all-target clippy, 계약 fixture 40개, 구조 검사, fmt·diff 검사가 통과했습니다. 공유 detection/loader 경로는 그대로입니다. 두 독립 리뷰어는 이 revision의 라우터·단일 모델 캐시 경계 전체를 승인했습니다.

`single_model_entry_from_state_with_cache`는 동기 helper입니다. #1838 시작 경로는 앱별 `Arc<CatalogProjectionCache>` 하나를 계속 유지하고 `tokio::task::spawn_blocking` 안에서 호출해야 합니다. 폴링마다 캐시를 새로 만들거나 HTTP handler에서 캐시 없는 편의 accessor를 호출하면 안 됩니다. 캐시 helper·provider 전환은 이 이슈에서 검증하며, 단일 모델 프로덕션 route의 offload·응답성 검증은 #1838 범위입니다. 라우터 HTTP adapter는 이미 메타데이터 획득을 offload합니다.
