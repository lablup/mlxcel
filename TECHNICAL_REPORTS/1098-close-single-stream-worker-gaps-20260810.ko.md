# 기술 보고서: PR #1098 - single-stream worker gap 보강

**작성일**: 2026-08-10
**상태**: 열린 PR 기준 작성 완료
**언어**: Rust, Markdown
**위험도**: Medium

## 요약

PR #1098은 single-stream 계열 서버 경계에서 남아 있던 세 가지 틈을 닫는다. 전용 worker의 늦은 queue admission, 선언된 이미지 수와 실제 resolve된 이미지 수의 불일치, 그리고 Florence-2의 `usage.prompt_tokens` 과소계산이다. 구현은 먼저 전용 worker와 공용 request-preparation 경로를 고쳤고, 이후 리뷰 하드닝에서 `/v1/completions` 와 native `/completion` 에 남아 있던 queue overload 구멍을 찾아 head commit `cd3af51e`에서 마저 닫았다.

결과적으로 overloaded single-stream family는 이제 SSE를 연 뒤가 아니라 HTTP 경계에서 일관되게 `503`으로 실패하고, 잘못 선언된 이미지 요청은 worker dispatch 전에 거절되며, Florence-2 사용량 집계는 실제 fused encoder 길이를 반영한다.

## 1. 문제 정의

`mlxcel-server`의 dedicated single-stream family는 메인 batched scheduler 경로를 거치지 않기 때문에, 자체 queue-depth reservation 규율이 필요하다. 이 PR 전에는 두 가지 문제가 있었다.

- DiffusionGemma, LLaDA-2 MoE, Florence-2는 queue-depth snapshot 이후에도 요청을 받아들일 수 있었고, 일부 route는 SSE를 연 뒤에야 saturation을 발견했다.
- 공용 chat media acquisition은 partial image resolution을 허용했다. 저수준 helper로서는 맞는 동작이지만, HTTP request-preparation 경계에서는 선언된 이미지 개수가 조용히 text-only 또는 partial multimodal 요청으로 바뀔 수 있어 안전하지 않았다.

Florence-2에는 별도의 정확성 문제도 있었다. `usage.prompt_tokens`가 실제 fused encoder 시퀀스를 반영하지 못했다. 이 모델은 projected image-feature token과 image placeholder filtering 이후 남는 prompt token을 함께 소비하므로, text 쪽만 세면 실제 작업량보다 적게 집계된다.

## 2. 초기 구현

첫 구현 커밋 `34b10437`은 세 가지 핵심 변경을 넣었다.

### 2.1 dedicated single-stream worker용 atomic queue reservation

`src/server/model_provider.rs`, `src/server/state.rs`는 shared queue-depth gauge 위에 RAII 기반 `SingleStreamQueueReservation`을 추가했다. DiffusionGemma, LLaDA-2 MoE, Florence-2는 이제 enqueue 전에 reserve하고 dequeue 또는 send 실패 시 release한다. 이로써 route-level admission과 observability가 "대충 본 큐 깊이"가 아니라 실제 pending depth를 반영하게 됐다.

그 다음 chat, Anthropic, Responses route를 streaming 작업 시작 전에 reserve하도록 연결해, 실패 모드가 "SSE를 연 뒤 overload 발견"에서 "route 경계에서 깔끔한 HTTP `503 Service Unavailable` 반환"으로 바뀌었다.

### 2.2 request preparation 경계의 공용 image-cardinality 검증

`src/server/chat_request.rs`는 tolerant media acquisition이 끝난 뒤 선언된 이미지 수와 실제 resolve된 이미지 수를 검증하도록 바뀌었고, `src/server/media.rs`는 그 검증을 공용 로직으로 노출한다. 이로써 내부 caller를 위한 tolerant helper는 유지하되, HTTP request boundary는 fail-closed가 된다.

provider도 internal caller를 위해 같은 guard를 반복 적용하며, 회귀 테스트는 잘못된 요청이 reject된 뒤에도 같은 live worker가 다음 정상 요청을 계속 처리함을 증명한다.

### 2.3 Florence-2 fused prompt accounting

`src/models/florence2/model.rs`는 `fused_prompt_len`을 추가하고 greedy generation 경로가 실제 fused encoder 길이를 반환하도록 바꿨다. 이 길이는 projected image token 수와 placeholder image token을 제거한 뒤 남는 prompt token 수의 합이다. 서버의 Florence-2 worker는 이 실제 fused 길이를 `usage.prompt_tokens`로 사용한다.

## 3. 리뷰 하드닝

리뷰는 route 경계의 마지막 불일치를 하나 더 찾았다. `/v1/completions` 와 native `/completion` 은 여전히 admission snapshot 뒤에 single-stream queue saturation이 드러날 수 있는 경로를 갖고 있었고, non-streaming error mapping도 `QueueFullError`를 generic server error로 뭉개고 있었다.

head commit `cd3af51e`는 이 구멍을 다음 방식으로 닫는다.

- `/v1/completions`, `/completion` 에서 SSE를 열기 전에 single-stream queue slot을 reserve한다.
- 두 streaming path를 reserved generation entry point로 라우팅한다.
- streaming/non-streaming 모두에서 `QueueFullError`를 HTTP `503` / "All slots are busy"로 매핑한다.
- route 테스트를 확장해 chat, Responses, Anthropic-compatible messages, OpenAI Completions, native `/completion` 이 모두 같은 overload 동작을 보이도록 검증한다.

이 하드닝이 중요한 이유는 외부에 보이는 마지막 불일치를 없애기 때문이다. 이 보강이 없었다면 대부분의 single-stream family는 고쳐졌지만, 두 text-completions surface만 예전 overload semantics를 유지하게 된다.

## 4. Confirmed / Refuted Matrix

| Family | `--max-queue-depth` gap | `usage.prompt_tokens` gap | Declared/resolved image gap |
|------|------|------|------|
| Florence-2 | 확인됐고 수정됐다. Dedicated seq2seq serving은 이제 dispatch 전에 reserve하고, review-hardening completions 경로까지 포함해 saturation 시 route-level `503`을 반환한다. | 확인됐고 수정됐다. usage가 이제 image feature shape와 filtered prompt token을 합친 실제 fused encoder 길이를 보고한다. | 확인됐고 수정됐다. 공용 request preparation이 worker dispatch 전에 image-cardinality mismatch를 거절한다. |
| DiffusionGemma | 확인됐고 수정됐다. Dedicated diffusion serving이 같은 single-stream reservation 경로를 사용한다. | 반증됐다. 기존 집계가 이미 generation이 실제로 소비하는 expanded engine prompt slice를 사용하고 있었다. 동작 변경 대신 회귀 테스트를 추가했다. | 확인됐고 수정됐다. 공용 request preparation이 dispatch 전에 image-cardinality mismatch를 거절한다. |
| LLaDA-2 MoE | 확인됐고 수정됐다. Dedicated LLaDA serving이 같은 single-stream reservation 경로를 사용한다. | 반증됐다. LLaDA는 generation 전에 media를 거절하며, 이미 generator에 넘기는 동일한 text prompt slice를 집계하고 있었다. 동작 변경 대신 회귀 테스트를 추가했다. | 확인됐고 수정됐다. 공용 request preparation이 dispatch 전에 image-cardinality mismatch를 거절한다. |

## 5. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 27 |
| 추가 라인 | +1001 |
| 삭제 라인 | -82 |
| PR 커밋 수 | 2 |
| Head commit | `cd3af51e18709ef9f3308486838f398881012957` |

- dedicated single-stream worker용 shared atomic queue-slot reservation/release를 추가했다.
- chat, Responses, Anthropic/messages, OpenAI Completions, native `/completion` 에서 overload 처리를 pre-SSE reservation으로 이동했다.
- request-preparation 경계에 공용 image-cardinality 검증과 provider-level recovery coverage를 추가했다.
- Florence-2 usage accounting을 실제 fused encoder 길이로 고쳤다.
- supported models, audio preprocessing, block diffusion 관련 사용자 문서를 업데이트했다.

## 6. 검증

### 2026-08-10 로컬 검증

- `cargo fmt --check`: 통과.
- `git diff --check origin/main...HEAD`: 통과.
- `cargo clippy -p mlxcel --lib --tests -- -D warnings`: 통과.
- `cargo test -p mlxcel --lib server::`: 통과. 결과는 `1673 passed; 0 failed; 8 ignored`.
- `cargo test -p mlxcel --lib server::model_provider::tests`: 통과. `18 passed`.
- `cargo test -p mlxcel --lib server::chat_request::tests`: 통과. `76 passed`.
- `cargo test -p mlxcel --lib server::max_tokens_route_tests`: 통과. `7 passed`.
- `cargo test -p mlxcel --lib server::media::tests`: 통과. `36 passed`.
- `cargo test -p mlxcel --lib server::state::tests`: 통과. `17 passed`.
- `cargo test -p mlxcel --lib server::diffusion_worker::tests`: 통과. `13 passed`.
- `cargo test -p mlxcel --lib models::florence2::model::florence2_fusion_tests`: 통과. `38 passed`.

### PR #1098에서 확인한 hosted checks

- `Detect changes`: pass
- `crate versions`: pass
- `kernel dtype keys`: pass
- `cross-repo refs`: pass
- `cargo-deny`: pass
- `cargo-fmt`: pass
- `license/cla`: pass
- `MLX pin extraction`: skipped

### 정직한 unavailable gate 구분

- 이번 보고서 패스에서는 full CUDA qualification을 실행하지 않았다. 이 저장소의 의미 있는 CUDA gate는 hardware/backend 의존성이 강하므로, 여기서는 full-CUDA 통과를 주장하지 않는다.
- Florence-2, DiffusionGemma, LLaDA-2 real-checkpoint 검증은 로컬에서 불가능했다. 이 머신에 필요한 family checkpoint가 없어서 live checkpoint pass는 수행하지 않았고, 따라서 주장하지 않는다.

## 7. 핵심 기술적 교훈

가장 중요한 설계 선택은 queue admission을 "나중에 관측하는 값"이 아니라 reservation으로 바꾼 점이다. 그래야 route 동작, worker enqueue, queue-depth metric이 같은 상태 전이를 공유하고 race에서 서로 어긋나지 않는다.

두 번째 교훈은 tolerant helper의 경계 배치다. media acquisition 자체는 내부적으로 관대해도 되지만, request boundary에서는 선언 수와 resolve 수를 명시적으로 검증해야 한다. 그렇지 않으면 best-effort helper가 요청 의미를 조용히 바꿔 버린다.

## 8. 관련 작업

- PR #1098: https://github.com/lablup/mlxcel/pull/1098
- Issue #1086: https://github.com/lablup/mlxcel/issues/1086
