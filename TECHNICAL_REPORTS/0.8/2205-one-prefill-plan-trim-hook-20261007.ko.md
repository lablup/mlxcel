# 기술 보고서: PR #2205 - CLI와 서버가 함께 쓰는 prefill 계획 하나와 trim 훅 하나

**작성일**: 2026-10-07

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust, Markdown

**위험도**: 중간.
- 이제 CLI와 서버의 모든 prefill이 계획 하나를 실행하고, 서버의 기본 prefill 청크가 512에서 2048로 바뀐다.
- 실제 체크포인트 세 개에서 greedy 출력은 변경 전후가 같다.
- 새 기본값이 동시 decode 지연에 주는 영향은 아직 측정하지 않았고, epic의 마지막 측정에서 다룬다.

## 요약

epic #2166의 Phase 3(#2170)이다.
- **계획 하나**: 이제 `mlxcel_core::prefill_plan::PrefillPlan` 한 곳에서 프롬프트를 어떻게 forward로 나눌지 정한다. 재사용할 prompt-cache 접두사, 대화 이력 경계에서의 분할, 청크 분할, 타일 패딩, 패딩 trim이 모두 여기서 정해진다. CLI와 서버의 prefill 경로는 그 계획의 조각을 실행하기만 한다.
- **trim 훅 하나**: 따로 있던 두 훅을 `LanguageModel::trim_state(Option<SequenceId>, excess)` 하나로 바꿨다.
- **청크 크기 하나**: ADR 0007의 측정에 따라 2048 하나로 정했다.
- **prompt-cache 차이 설명**: Phase 0에서 발견한 캐시 적중과 미적중의 출력 차이를 원인(prefill 분할)까지 밝히고, 명시적인 불변식으로 범위를 정했다.

## 1. 문제 정의

- **prefill 결정이 두 곳에 있었다.** CLI는 2048, 서버는 512로 청크를 나눴고, 패딩과 trim도 양쪽에 따로 짜여 있었다.
- **trim 훅이 두 개였다.** 모델 API에 CLI용 `trim_internal_caches`와 서버용 `trim_sequence_state`가 따로 있어서, 모델 계열마다 둘 다 구현해야 했다. #2140이 그 때문에 생긴 문제다.
- **prompt-cache 차이**: Phase 0 패리티 기준선에서, 서버의 prompt-cache 적중 실행이 같은 프롬프트를 캐시 없이 돌린 결과와 달랐다. Qwen3-1.7B greedy는 0번째 토큰부터 달랐다.

## 2. 변경 요약

- **`PrefillPlan`**
  - `new`/`with_prefix(prompt_len, adopted, boundary, chunk, PrefillCaps)`가 순서대로 `PrefillPiece { range, padded_len }`를 만든다.
  - 대화 이력 경계 구간은 패딩 없는 조각 하나로 두고, 청크는 경계부터 다시 센다.
  - 마지막 조각은 모델과 입력이 허용할 때만 타일에 맞춰 패딩한다.
  - `reproduces`와 `split_points`는 캐시 적중이 미적중 결과를 재현할 수 있는 조건을 나타낸다.
- **CLI**: `prefill_prompt_last_logits`, `generate_with_stats` 안의 복사본, 임베딩 변형 두 개가 모두 하나의 실행기로 계획의 조각을 실행한다.
- **서버**
  - `planned_prefill.rs`가 전체 prefill, 청크 prefill, 배치 prefill, handoff prefill의 조각을 실행한다. 계획은 tick마다 시퀀스에서 다시 만들고, 진행 위치는 `prefill_offset`이 가리킨다.
  - 대화 이력 경계의 snapshot은 경계 조각을 실행한 직후에 forward를 더 돌리지 않고 넣는다.
  - Gemma 4의 `mtp_prefill_ranges`와 paged 블록 예약도 계획을 쓴다.
- **`trim_state`**: 자체 상태를 가진 모델 계열 일곱 개가 구현하고, VLM wrapper는 전달하며, `LoadedModel`은 위임한다. `rewind_decode_appends`는 별개의 정확한 되감기 계약으로 그대로 둔다.
- **청크 정책**: `prefill_chunk_len()`은 `MLXCEL_PREFILL_CHUNK`를 읽고, 없으면 2048을 쓴다. 이제 서버 바이너리 두 개 모두에서 `--prefill-chunk-size`의 기본값이기도 하다.
  - 메모리 추정의 활성화 항(`activation_prefill_tokens()`)도 이 값을 따른다.
  - 배치 prefill 토큰 예산은 행마다 512로 상한을 두어 묶음 예산을 4096으로 유지한다.

## 3. 기술적 선택과 그 이유

**캐시 차이의 원인은 재사용이 아니라 분할이다.** `MLXCEL_SDPA_DETERMINISTIC=1`에서 미적중 실행을 캐시된 길이에서 청크로 나누자 두 실행이 같아졌다.
- Qwen3-1.7B: 45에서 분할
- Llama-3.2-1B: 71에서 분할

원리는 이렇다. dense KV를 쓰는 미적중 실행은 52행 또는 75행을 forward 한 번으로 처리한다(타일형 qmm). 적중 실행은 7행 또는 4행 꼬리만 forward하는데, 이 크기는 행 단위 qmv 커널(`M*B < 8`)과 다른 attention 타일링을 탄다. 두 실행은 합산 순서가 달라서, 점수가 거의 같은 토큰이 뒤바뀐다.

**분할을 강제하지 않고 경계를 문서로 정했다.** 적중 실행은 재사용하는 접두사가 미적중 실행의 분할 지점에서 끝나고, 그 행들을 같은 조각이 썼을 때 미적중 결과를 정확히 재현한다. 모든 모델 계열을 대화 이력 경계에서 강제로 나누면 하네스 결과는 같아지지만, 세 가지 이유로 택하지 않았다.
- 모든 cold chat prefill에 forward가 한 번 더 든다.
- prompt cache가 한 턴짜리 출력까지 바꾸게 된다.
- 프롬프트 전체를 다시 보내는 경우는 어차피 해결되지 않는다.

**청크 크기는 2048 하나다.** Phase 0에서 2048이 8192토큰 기준 TTFT를 8~29% 낮췄다. 리뷰에서 512를 말없이 가정하던 곳 세 군데를 찾아 고쳤다.
- `mlx_server`의 플래그 기본값
- 메모리 추정의 `clamp(1, 0)` panic
- 배치 prefill 예산

## 4. 검증

- **단위 테스트**
  - `prefill_plan` 11, mlxcel-core `prefill` 76, `generate::` 46, `decode_finish` 9.
  - 루트 크레이트는 모듈을 하나씩 돌렸다: `scheduler_prompt_cache_plan_tests` 3(청크 카운터와 진행 프레임 경우 포함), `server::batch::scheduler::tests` 94, `scheduler_prompt_cache_tests` 31, `prefill_span_coverage_tests` 3, `finish_step_tests`, `speculative_burst_tests`, `prefill_cohort`, `cli_input`, `memory_estimate`, `gemma4_mtp_target`.
  - 타일 패딩 보호 감사를 포함한 `vlm_wrapper_capability_delegation`.
- **실제 체크포인트**: qwen3-1.7b-4bit, llama-3.2-1b-instruct-4bit, lfm2-350m-8bit에서 `MLXCEL_SDPA_DETERMINISTIC=1`로 `mlxcel-engine-parity`를 돌렸다.
  - 모든 비교 쌍의 결과가 변경 전, 변경 후, main 병합 후 모두 같았다.
  - CLI와 서버의 토큰 열은 변경 전 커밋과 같았다.
- **알려진 불안정성**: `server::batch::scheduler` 필터 전체를 한 번에 돌리면 CUDA graph capture에서 가끔 중단된다. 모듈을 하나씩 돌리면 모두 깨끗하다.

## 5. 남은 위험

- **동시 decode 지연을 재지 않았다.** 2048 기본값에서 동시 decode 스트림의 토큰 간 지연은 epic 마지막 측정으로 넘겼다. prefill 조각이 길어지면 decode 차례가 최대 네 배까지 늦어진다.
- **대화 이력 구간은 청크로 나뉘지 않는다.** snapshot 계열의 이력 구간은 청크 크기와 상관없이 forward 한 번으로 처리된다. 이전부터 있던 동작이다(#1143).
- **환경 변수 하나가 양쪽을 움직인다.** `MLXCEL_PREFILL_CHUNK`가 이제 서버 기본값도 정하고, 메모리 추정은 명시적인 `--prefill-chunk-size`를 보지 못한다.
- **수치 경계 테스트가 없다.** 적중과 미적중의 차이에 수치 경계를 거는 테스트는 없다. 경계는 문서화한 불변식과 측정표가 대신한다.

## 6. 학습 포인트

- **분할 방식도 수치 계산의 일부로 봐야 한다.** 같은 토큰을 올바르게 처리한 두 prefill도 나누는 방식이 다르면 낮은 비트가 달라진다. 커널 선택이 행 수에 따라 달라지기 때문이다.
- **기본값을 바꿀 때는 그 값에 암묵적으로 기대던 곳을 모두 찾아야 한다.** 리뷰에서 고친 네 건 가운데 세 건이 이름을 밝히지 않고 512를 가정하던 곳이었다.

## 7. 관련 항목

- epic #2166, 이슈 #2170.
- ADR 0007, ADR 0005.
- #2201(종료 처리 통합, 이 브랜치에 병합됨).
- #2140, #1143, #2185.
