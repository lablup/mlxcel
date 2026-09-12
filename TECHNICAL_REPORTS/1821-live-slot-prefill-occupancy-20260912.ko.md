# 기술 리포트: PR #1821 - fix: track live slot occupancy during prefill

**날짜**: 2026-09-12
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클(구현 리뷰, 보안·성능 리뷰)
**상태**: 완료(pinned b10621 실행 파일이 없어 직접 비교는 수행하지 못함)
**언어**: Rust, JSON, Markdown
**위험도**: 중간(여러 라우트의 관측 경로와 공유 슬롯 상태 변경; 추론 산술은 바뀌지 않음)

---

## 요약

PR #1821은 `GET /slots`가 모든 생성 API에서 스케줄러 수락부터 prefill, decode, 완료까지 요청 상태를 반영하게 한다. 스케줄러가 이미 내보내는 prefill 진행 이벤트를 라우트 소유 슬롯 핸들에 전달하고, 실시간 및 완료 후 카운터를 llama-server b10621 의미에 맞추며, `--ctx-size 0`이 제한을 체크포인트에 위임할 때도 양의 유효 컨텍스트 창을 보고한다.

리뷰에서는 세 메타데이터 엔드포인트가 체크포인트 설정을 반복해서 읽던 문제도 제거했다. 이제 유효 컨텍스트 창은 `AppState` 생성 시 한 번 계산되어 `/slots`, `/props`, `/v1/models`에서 재사용된다.

---

## 문제 정의

기존 슬롯 레지스트리는 디코드된 텍스트 조각이 도착한 뒤에야 요청을 슬롯에 묶었다. 따라서 긴 prefill 중에는 서버가 유휴 상태처럼 보였고, 스케줄러가 이미 처리 중인 요청과 아직 대기 중인 요청을 구분할 수 없었다. 또한 완료 후 prompt/completion 카운터가 b10621과 다른 의미를 썼고, 생성 경로는 양의 모델 기반 제한을 사용하면서도 메타데이터는 `n_ctx: 0`을 반환했다.

이 차이는 `/slots`를 admission, 포화 상태, 진행률 관측에 쓰는 운영자에게 직접 영향을 줬다.

- prefill 동안 `is_processing`이 false로 남았다.
- prompt cache 및 처리 토큰 카운터가 prompt 평가 중 증가하지 않았다.
- decode 중과 완료 후 `n_prompt_tokens`가 prompt와 수락된 decode 토큰의 합이 되지 않았다.
- `--ctx-size 0`에서 `/slots`, `/props`, `/v1/models`가 실제 생성 컨텍스트와 달랐다.

---

## 변경 요약

- 토큰과 logprob 전달은 유지하면서 모든 `GenerateEvent::Prefill` 관측을 전달하는 provider drain 변형을 추가했다.
- Chat Completions, text completions, Responses, Anthropic messages, native completion, ASR streaming 분기의 streaming/non-streaming 경로를 각 라우트의 `SlotHandle`에 연결했다.
- 첫 스케줄러 진행 신호에서 요청을 슬롯에 묶고, 스케줄러 큐에서 기다리는 요청은 계속 미할당 상태로 뒀다.
- `SlotHandle::on_prefill_progress`를 추가하고 실시간/최종 카운터를 b10621 의미로 갱신했다. 요청에서 유래한 값은 prompt 길이에 맞춰 제한하거나 saturating 연산으로 결합한다.
- native `return_progress` 프레임과 슬롯 계측을 독립시켰다. 모든 관측은 슬롯을 갱신하지만 클라이언트 진행 프레임은 요청했을 때만 보낸다.
- 유효 슬롯별 컨텍스트를 서버 상태 생성 시 한 번 계산하고 세 메타데이터 표면에서 재사용한다.
- b10621 호환성 manifest, 환경 변수 문서, 호환성 문서, 라우트/상태/provider 집중 테스트를 갱신했다.

최종 PR은 두 커밋에 걸쳐 20개 파일, 503줄 추가, 49줄 삭제로 구성된다.

---

## 기술적 선택과 그 이유

### 스케줄러 진행을 슬롯 결합 경계로 사용

HTTP 수락 시 `SlotRegistry::begin`에서 즉시 결합하면 아직 큐에서 기다리는 요청이 슬롯을 차지한다. 반대로 디코드 텍스트를 기다리면 너무 늦다. `GenerateEvent::Prefill`은 스케줄러가 요청을 수락한 뒤 처음 나오는 신호이므로, 관측 레지스트리가 admission control을 떠맡지 않으면서 정확한 경계를 제공한다.

prefill 관측을 내지 않는 backend를 위해 기존 첫 토큰 갱신은 fallback으로 남겼다. `SlotHandle`의 RAII 해제도 그대로라서 오류, 취소, streaming task drop 때 결합된 슬롯이 유휴 상태로 돌아간다.

### Provider drain에서 진행 이벤트 전달

Provider는 streaming과 non-streaming 라우트가 함께 쓰는 동기 이벤트 drain 경계를 이미 소유한다. 여기에 타입이 있는 prefill observer를 추가하면 이벤트 순서를 보존할 수 있다. 진행 갱신은 첫 토큰과 최종 결과보다 먼저 적용되며, 라우트가 스케줄러 내부 상태를 직접 읽지 않아도 된다.

### 유효 컨텍스트를 `AppState`에 캐시

첫 구현은 올바른 체크포인트 해석 함수를 재사용했지만 메타데이터 요청마다 호출했다. `/slots`는 자주 polling되므로 관측 엔드포인트가 매번 `config.json`을 읽고 파싱하게 됐다. 시작 시 해석하면 같은 fallback 순서—명시한 슬롯별 값, 체크포인트 컨텍스트, 4096—를 유지하면서 요청 처리는 필드 읽기 하나로 줄어든다.

---

## 기술적 검토 사항

### 보안과 동시성

- Prompt 및 생성 텍스트는 기존 debug 또는 slot-save opt-in에서만 보존한다. 새 observer는 기본적으로 카운터만 저장한다.
- 슬롯 갱신은 크기가 제한된 서버 소유 상태만 변경하며, path, command, 새 역직렬화 형식을 받지 않는다.
- `SlotHandle`과 `SlotRegistry`는 일관된 handle-then-registry 잠금 순서를 유지하고 새 역순 획득을 추가하지 않았다.
- Cache 및 prompt 합계는 prompt 길이에 맞춰 제한하며, 외부 영향이 있는 카운터를 합칠 때 saturating 연산을 쓴다.

리뷰 후 남은 CRITICAL 또는 HIGH 보안 문제는 없다.

### 성능

Prefill 관측은 기존 슬롯 mutex 아래에서 고정 크기 카운터만 갱신한다. 확인된 성능 문제는 `--ctx-size 0`에서 반복되는 체크포인트 설정 I/O 하나였다. 커밋 `ac8162bf`가 이 값을 `AppState` 생성 시 한 번만 계산해 저장한다.

---

## 검증

- 주 구현 커밋에서 전체 워크스페이스 게이트 `cargo test --workspace --profile test-fast --features metal,accelerate`가 실패 없이 성공했다.
- 최종 리뷰 커밋:
  - `RUSTC_WRAPPER= cargo test --profile test-fast ctx_size_zero` — 일치하는 라우트 테스트 3개 통과.
  - `RUSTC_WRAPPER= cargo clippy --all-targets --features metal,accelerate -- -D warnings` — 통과.
  - `cargo fmt --check`, `git diff --check` — 통과.
  - 최종 GitHub CI에서 crate versions, kernel dtype keys, license headers, compatibility manifest, cross-repository references, cargo-deny, cargo-fmt, cargo-clippy, OpenXLA feature compile 통과.
- Prefill 관측 전달과 대기 요청 격리를 포함한 슬롯/provider/manifest 집중 테스트가 통과했다.
- 실제 체크포인트 `models/gemma-3-1b-it-4bit`:
  - `/props`, `/v1/models`, `/slots`가 `n_ctx: 1024`를 보고했다.
  - 6토큰 prompt와 2개 decode 토큰 뒤 `/v1/completions`가 `n_prompt_tokens: 8`을 보존했다.
  - Chat Completions, completions, Responses, Anthropic messages, native completion의 streaming/non-streaming 1토큰 요청이 완료됐다.

Pinned llama-server b10621 실행 파일이 로컬에 없고 b10883 산출물만 있어 정확한 upstream 실행 비교는 반복하지 못했다. 다른 버전을 대체 기준으로 쓰지 않고 이 제한을 PR에 명시했다.

---

## 학습 포인트

- 관측용 슬롯은 HTTP 수신 시점이나 첫 가시 텍스트가 아니라 스케줄러의 첫 서비스 이벤트에서 결합해야 한다.
- 스케줄러 이벤트 하나를 protocol 출력과 내부 관측이 함께 쓸 수 있지만, 소비 조건은 독립적이어야 한다. 슬롯 계측은 무조건 수행해도 native `return_progress`는 계속 opt-in이다.
- 올바른 resolver를 재사용해도 그 함수가 I/O를 수행하면 충분하지 않다. 서버 수명 동안 고정된 값은 특히 polling 엔드포인트에 노출될 때 생성 시점에 구체화해야 한다.

---

## 후속 조치

- Unified KV budget과 auto-`--parallel` 컨텍스트 geometry는 의도적으로 범위 밖이며 issue #1815가 추적한다.
- Pinned b10621 실행 파일을 확보하면 long-prompt parallel 1/2 비교를 다시 수행한다. 이후 버전을 조용히 대체 기준으로 사용하지 않는다.

