# 기술 보고서: PR #1268 - 확정된 MTP 판정을 위한 지원되는 읽기 인터페이스

## 요약

`MtpPolicy`(이슈 #333)는 실제 기계에서 페어링별 MTP 판정을 확정하고 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/<해시>.json`에 `PolicyHint`로 남긴다. 이 판정은 "여기서 이 페어링에 MTP가 도움이 되는가"에 대한 유일한 기계별 답이고, 호스트 애플리케이션이 사용자에게 보여주고 싶어 하는 값이다. API가 없어서 Backend.AI GO(lablup/backend.ai-go#4441)는 다른 프로세스에서 mlxcel의 비공개 캐시 파일을 파싱하는 리더를 이미 출하했다.

이 PR은 이슈가 선호한 Option A, `GET /v1/internal/mtp-policy`를 추가한다. 캐시 파일은 건드리지 않는다. 이미 의존하는 소비자가 있기 때문이다.

## 1. 문제

온디스크 형식에 결합하는 것은 어느 쪽 프로젝트도 감지하지 못하는 방식으로 취약하다. `HINT_VERSION`은 스키마나 판정 의미론이 바뀔 때마다 올라가고, `HINT_SUBDIR`은 비공개 상수이며, `PolicyHint`는 `pub(crate)`다. 셋 중 무엇이든 패치 릴리스에서 움직일 수 있다. 소비자는 크래시가 아니라 빈 화면으로 degrade하는데, 진단 관점에서는 그쪽이 더 나쁘다. 사용자는 아무것도 못 보고 어디에서도 오류가 나지 않는다.

이슈는 파일 형식이 표현할 수 없는 상태도 요구했다. 파일 부재는 "아직 프로파일링 중", "MTP 페어링 없음", "캐시 루트가 엉뚱한 곳으로 풀렸음"을 한꺼번에 뜻한다.

## 2. 기술적 판단

### 2.1 정책에 락을 걸지 않고 observability로 발행한다

`MtpPolicy`는 단일 스레드로 문서화돼 있다. 스케줄러가 워커 스레드에서 소유하며 락이 필요 없다. axum 핸들러는 다른 곳에서 돈다. 이걸 `Arc<Mutex<..>>`로 감싸면 진단용 엔드포인트를 위해 디코드 핫 경로에 락을 놓는 셈이다.

대신 스케줄러가 이미 공유되고 있는 `Arc<BatchObservability>`에 `MtpPolicySnapshot`을 발행하고 핸들러가 읽는다. 발행 시점은 워커 시작 때, 그리고 상태가 아직 움직일 수 있는 동안뿐이다. 버스트 경로는 `record_b1_sample` **호출 전에** 잡은 `was_profiling` 가드 아래에서 재발행하므로 프로파일링에서 확정으로 넘어가는 전이 자체가 포착된다. 호출 뒤에 잡았다면 바로 그 전이를 놓쳤을 것이다. 확정되거나 강제된 뒤의 정상 상태는 비용이 0이다.

### 2.2 `forced`를 독립 상태로 보고한다

이슈는 상태 세 가지를 지목했다. 코드에는 넷이 있다. `PolicyState::Forced`는 `MLXCEL_ENABLE_MTP_B1`이 결정을 고정했고 이 기계에서는 아무것도 측정되지 않았다는 뜻이다. 이걸 `settled`에 뭉개면 환경변수에서 온 값을 두고 소비자가 "이 기계가 MTP를 이득이라고 측정했다"고 렌더링하게 된다. 그래서 `forced`는 `verdict: null`, 수용률 없음, 샘플 수 없음으로 보고한다.

관련 불변식은 재시작을 건너서도 유지된다. `from_parts`는 저장된 힌트보다 핀을 먼저 확인하고, `record_b1_sample`은 `Profiling`이 아닌 모든 상태에서 조기 반환하므로, 강제 실행은 나중에 핀 없이 뜬 프로세스가 측정된 판정으로 되읽을 힌트를 결코 남기지 않는다.

### 2.3 `unavailable`에 사유를 싣는다

`no_mtp_dispatch`, `adaptive_disabled`, `worker_not_ready`는 서로 다른 상황이고, 뭉치면 이슈가 지적한 바로 그 모호함을 재생산한다. `adaptive_disabled`에서도 정적 게이트(또는 운영자 핀)가 무엇을 결정했는지를 `mtp_enabled`로 계속 보고하므로, 답이 정직하기만 한 게 아니라 쓸모가 있다.

### 2.4 확정 시 반올림된 수용률을 발행한다

`PolicyHint::new`는 저장 전에 소수 두 자리로 반올림한다. 이제 확정 상태는 원시 몫이 아니라 `hint.acceptance_rate`를 가져가므로, 파일에서 엔드포인트로 이전하는 소비자가 두 출처의 값이 반올림 한 단계만큼 어긋나는 것을 볼 수 없다. `store.save`가 실패해도 값은 힌트에서 가져온다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/server/routes/mtp_policy.rs` | 신규 핸들러, `MtpPolicyResponse`, `MTP_POLICY_SCHEMA_VERSION = 1`. 매핑은 `cache::build_stats_response`를 따라 순수 헬퍼로 분리 |
| `src/server/batch/mtp_policy.rs` | `MtpPolicyStatus`, `MtpPolicyUnavailableReason`, `MtpPolicySnapshot`, `snapshot()`, `is_profiling()`. `Settled`는 반올림된 수용률과 샘플 수를 싣는 구조체 variant로 |
| `src/server/batch/observability.rs` | `mtp_policy: Mutex<Option<MtpPolicySnapshot>>`와 접근자, `ObservabilitySnapshot`의 필드 |
| `src/server/batch/scheduler.rs` | `with_mtp_policy`에서 발행, `was_profiling` 가드 아래 재발행 |
| `src/server/model_worker.rs` | 레거시 `--no-batch`와 OpenXLA 워커가 시작 시 `no_mtp_dispatch` 발행 |
| `src/server/app.rs`, `routes/mod.rs`, `batch/mod.rs` | 라우트 등록과 재수출 |
| `docs/mtp-policy-api.md` | 신규 레퍼런스: 본문, 상태/사유 표, 버저닝과 호환성 정책 |

## 4. 리뷰 지적사항

독립 리뷰 두 번이 돌았다. CRITICAL/HIGH는 없었고, 둘 다 핵심 성질을 확인했다. 운영자 핀은 측정된 판정으로 표면화될 수 없고, `record_b1_sample`이 `PolicyState`의 유일한 변경자이며 호출 지점이 하나라 발행된 스냅샷이 낙후될 수 없으며, `Mutex`는 fail-closed다(poisoning은 낙후된 판정이 아니라 `unavailable`로 드러난다). `/health` 추가는 Prometheus 텍스트 페이로드를 바이트 단위로 그대로 둔다.

두 번째 커밋(`0f624651`)이 지적사항을 처리했고, 전부 동작이 아니라 정확성에 관한 것이다.

- 새로 공개된 두 enum에 `#[non_exhaustive]`. `docs/mtp-policy-api.md`가 라벨 집합이 늘어난다고 약속하고, 저장소는 이미 이 속성을 약 94곳에서 쓴다.
- 레거시 `--no-batch`와 OpenXLA 워커는 `with_mtp_policy`를 호출한 적이 없어서, 정직한 답이 `no_mtp_dispatch`인데도 엔드포인트가 프로세스 수명 내내 `worker_not_ready`로 답했다. 이제 둘 다 시작 시 발행한다.
- 신규 문서의 거짓 진술 네 건: 엔드포인트는 `--node-role router` 프런트엔드에는 마운트되지 않는다. `--num-draft-tokens`는 정책 키를 다시 만들지 않는다(`--draft-block-size`가 한다. 이 거짓 주장은 소스 주석에서 복사된 것이라 그 주석도 함께 고쳤다). `PolicyHint`는 이미 `hardware`를 싣고 있어 "추가로"라는 서술이 틀렸다. `adaptive_disabled` 설명이 실제로는 운영자 핀이 결정한 경우에 정적 게이트의 공으로 돌렸다.
- `profiling` 상태의 정확성 한계를 암묵이 아니라 명시로 적었다. `with_mtp_policy`는 디스패치가 **설정되어 있다는** 사실로 `profiling`을 발행하지만 버스트에는 이후 런타임 게이트가 더 있다. 그것들이 모든 버스트를 거절하면 엔드포인트는 프로세스 수명 내내 `samples: 0`인 `profiling`을 보고하고, 그 구성에서 `mtp_enabled: true`는 사실이 아니다. 수정에 설계 판단이 필요해서 고치는 대신 문서화하기로 했다.

## 5. 검증

GB10(DGX Spark, CUDA sm_121, Linux aarch64), MLX 핀 `9a795735`에서 실측.

- `9ff0a3ad`에서 `make verify-test-cuda`: **8188 통과, 0 실패, 310 무시**, exit 0. `main`의 `a940c737`(8171) 대비 +17이고, diff가 `#[test]`를 정확히 17개 추가하고 하나도 제거하지 않으므로 수치가 그럴듯한 수준이 아니라 실제로 맞아떨어진다.
- `0f624651`에서 `make verify-test-cuda` 재실행: PR 스레드에 기록.
- `cargo fmt --all -- --check`, `cargo clippy --lib --tests --features cuda -- -D warnings`: 클린.

MTP 페어링이 적재된 실제 서버에 대고 엔드포인트를 호출해 보지는 않았다. 라우트 배선은 `create_app`을 통해 컴파일 검증되고, 응답 매핑과 발행/읽기 이음매는 단위 테스트로 덮인다.

## 6. 관련 작업

- #1257: 이 PR이 닫는 이슈. lablup/backend.ai-go#4441이 이 인터페이스로 대체될 임시 캐시 리더를 가진 소비자다.
- #333: 판정을 만들어내는 적응형 정책.
- 온디스크 힌트 형식은 의도적으로 그대로다. 본문의 `schema_version`은 1에서 시작하고 파일의 `HINT_VERSION = 3`과 독립이다. 둘을 같이 읽는 소비자는 서로 다른 버전 번호 둘을 보게 되므로 문서에 명시했다.
