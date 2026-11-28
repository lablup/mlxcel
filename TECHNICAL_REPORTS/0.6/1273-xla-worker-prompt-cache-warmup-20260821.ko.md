# 기술 보고서: PR #1273 - OpenXLA 워커의 PromptCacheWarmup 처리

## 요약

`main`이 `--features xla-iree`로 컴파일되지 않았다. PR #1154가 `ModelRequest::PromptCacheWarmup` variant를 추가하면서 자신이 아는 워커 셋은 갱신했지만 OpenXLA 워커는 빠뜨려, `match`가 비망라 상태로 남았다. 수정은 작업을 폐기하는 다섯 줄짜리 arm이다.

변경 자체는 사소하다. 기록할 가치가 있는 것은 컴파일 에러가 `main`에 탐지되지 않은 채 앉아 있었다는 사실과, 저장소의 CI가 이 부류의 파손을 아예 볼 수 없는 이유다.

## 1. 문제

```text
error[E0004]: non-exhaustive patterns: `model_provider::ModelRequest::PromptCacheWarmup { .. }` not covered
  --> src/server/batch/xla_worker_admission.rs:482:15
```

`src/server/batch/mod.rs`가 `xla_preprocess`와 `xla_worker`를 `#[cfg(feature = "xla-iree")]` 뒤에 두고, 어떤 CI 워크플로도 XLA feature를 켜지 않는다. `ci.yml`은 `deny`, `fmt`, crate-version, kernel-dtype-key, MLX-pin, cross-repo-ref 잡을 돌리며 워크스페이스 `cargo check`가 없다. `pipeline-parallel-ci.yml`은 기본 feature로 clippy를 돌린다. `nightly-verify.yml`은 `metal,accelerate`로 clippy를 돌린다. 어느 것도 이 모듈을 컴파일하지 않으므로, #1154에서 green을 보고한 모든 체크에게 이 파손은 보이지 않았다.

PR #916을 `main` 위로 리베이스하고 그 브랜치를 XLA feature로 빌드했을 때에야 드러났다.

## 2. 기술적 판단

### 2.1 전달하거나 오류를 내지 않고 폐기한다

프롬프트 캐시는 `BatchScheduler`만 소유한다. OpenXLA 워커에는 워밍할 스냅샷 상태가 없으므로 전달할 대상도 없고, 요청이 도착했다고 해서 잘못된 것도 없다. `diffusion_worker.rs`와 `florence2_worker.rs`가 같은 이유로 같은 결론에 이미 도달했으므로 이 arm도 그들과 맞췄다. 빈 블록을 조용히 두지 않고 이유를 주석으로 남긴 것까지 포함해서다.

### 2.2 admission 상태를 건드리지 않는다

이 arm은 의도적으로 아무것도 하지 않는다. `PromptCacheWarmup`은 `response_tx`도 큐 예약도 담지 않으므로 기다리는 호출자가 없고 해제할 게이지도 없다. "정리"한답시고 대기 중인 이미지나 오디오 상태를 건드리는 것은 variant의 계약이 존재하지 않는다고 말하는 작업을 지어내는 셈이다.

### 2.3 PR #916에서 분리

이 문제는 #916을 리베이스하다 발견됐고 처음에는 그 브랜치에서 무관한 re-export 게이트와 묶여 수정됐다. #916은 큰 기능 PR이고 이것은 다른 브랜치들도 필요로 하는 `main` 컴파일 수정이라 분리했다. 분리하면서 오귀속도 드러났다. 함께 묶여 있던 re-export 게이트는 `main`의 결함이 전혀 아니었고 #916 자신이 도입한 것이어서, 그 브랜치에 남았다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `src/server/batch/xla_worker_admission.rs` | `XlaServeWorker::handle`에 `ModelRequest::PromptCacheWarmup { .. }` arm 추가, 이유를 기록하고 작업 폐기 |

## 4. 리뷰 지적사항

지적사항 없다. 크레이트를 컴파일되게 만드는 것 외에 동작 표면이 없는 match arm 하나다.

## 5. 검증

| 명령 | 이전 | 이후 |
| --- | --- | --- |
| `cargo check --features cuda,xla-iree --lib` | E0004 | 통과 |
| `cargo check --features cuda --all-targets` | 통과 | 통과, 무경고 |

둘 다 이 브랜치 단독으로, 작업 트리를 공유하는 다른 빌드 없이 실행했다. 이 세부가 중요했다. 이번 세션의 앞선 검증 시도는 두 스크립트가 같은 트리에서 서로 다른 리비전을 체크아웃하며 동시에 돌아 거짓 실패를 냈다.

## 6. 관련 작업

이슈 #1270이 이 문제의 지속적인 절반, 즉 CI에서 XLA feature 조합을 아무도 컴파일하지 않는다는 사실을 추적한다. 이 PR의 `Closes` 키워드로 닫혔다가 다시 열리고 제목이 바뀌었다. 네 개 수용 기준 중 첫 번째만 여기서 충족됐기 때문이다. 나머지 셋이 CI 잡 자체다.

이 커버리지 공백은 이제 결함 두 개를 `main`에 통과시켰다. 이 건과, `cargo check`가 링크를 하지 않아 잡을 수 없는 #1274의 링크 실패다. 이 한 쌍이 그 잡을 만들어야 한다는 근거다.
