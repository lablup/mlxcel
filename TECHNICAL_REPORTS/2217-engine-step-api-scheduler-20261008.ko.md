# 기술 보고서: PR #2217 - 배치 기반 엔진 step API와 그 위의 BatchScheduler

**작성일**: 2026-10-08

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust, Markdown(ADR, 문서)

**위험도**: 높음. 서버의 모든 forward, 샘플링, 종료 판정이 새 모듈을 거치고, Gemma 3과 Llama 4의 attention 경로를 다시 썼다. 세 모델 계열에서 실제 체크포인트 패리티가 그대로이고 실제 요청 두 개를 묶은 배치도 각 요청을 혼자 돌린 결과와 같았지만, 서빙되는 모든 요청의 hot path가 바뀐다.

## 요약

`BatchScheduler`는 무엇을 돌릴지 정하는 일과 그것을 실제로 돌리는 일을 함께 했고, 열두 곳 남짓에서 모델 forward와 샘플러를 직접 호출했다. epic #2166은 이 실행 부분을 CLI(Phase 5)와 `mlxcel generate`(Phase 6)가 재사용할 수 있어야 한다. Phase 4b(#2172)인 이 PR은 `mlxcel_core::engine`을 추가한다. `Engine<M>`은 모델과 `CachePool`을 소유하고 `open`, `prefill`, `step`, lookahead 파이프라인용으로 나눈 `submit`/`finish_rows`, `unwind_appends`, `close`를 제공한다. scheduler는 admission, tick 정책, preemption, prompt cache, handoff, speculative burst 생성기를 그대로 맡고, 모델은 엔진을 통해서만 구동한다. 시퀀스 하나는 행이 하나인 배치이고, decode 진입점은 `step` 하나뿐이다. 이 PR은 #2210이 시작한 attention 이전(Gemma 3, Llama 4, 모델 소유 헬퍼)도 마무리하고 `DecodeBatchContext`를 없앤다.

## 1. 문제 정의

- 모델 실행이 scheduler 안에 있어서, CLI가 이를 재사용하려면 scheduler까지 끌고 와야 했다.
- 서버의 B=1과 CLI의 단일 시퀀스는 서로 다른 코드였다. scheduler의 `decode_single_step`과 배치 decode가 각자 forward와 샘플러를 불렀고, `CxxGenerator`에는 별도의 루프가 있었다.
- 세 계열(Gemma 3, Llama 4, `model_owned` 헬퍼)은 #2210이 다른 계열에서 이미 걷어 낸 `DecodeBatchContext`로 여전히 dense와 paged 커널을 골랐다.

## 2. 변경 요약

- **엔진(`src/lib/mlxcel-core/src/engine/`)**
  - `open(SequenceSpec)`, `adopt`, `adopt_paged`, `close`(기존 `release_sequence_caches`와 동일).
  - `prefill(&PrefillStep)`은 `PrefillPlan` 조각 하나를 실행한다. 조각의 입력과 mask는 공유 함수 `piece_input`이 만든다. `prefill_cohort`는 padding된 cohort를, `complete_prefill`은 첫 토큰 샘플링을, `trim_padding`은 padding 되돌리기를 맡는다.
  - `step(&StepBatch, &mut [StepRow<H>]) -> StepOutput`: forward를 돌린 뒤 행마다 mask, draw, `try_eval`, thinking override, logprobs, matcher, `finish_step`을 거친다. 행이 둘 이상이고 모든 행이 `shared_fused_params`를 통과하면 fused `[B, vocab]` draw 한 번으로 처리하고, readback 전에 `try_eval`로 평가한다.
  - `submit`과 `finish_rows`: lookahead용으로 나눈 형태다. `submit`은 forward를 `DecodeLookaheadAppendScope` 안에서 돌리고, `unwind_appends`는 pool을 잘라 내거나 모델 소유 계열이면 `rewind_decode_appends`를 부른다.
  - `StepRow`에는 행의 `RowSampler`, 설정, history, 예산, 그리고 `H: StepRowHooks`(logit mask, override, matcher, 첫 토큰 기록, `FinishHooks`)가 담긴다. `RowOutcome`에는 토큰, 선택적 `FinishCause`, 선택적 `RowError::{Structured, Eval, BatchEval}`이 담긴다.
- **scheduler**: `step_rows.rs`가 행을 만들고 결과를 반영한다. decode_tick, prefill, planned prefill, prompt cache, handoff가 엔진을 호출한다. `src/server/batch/scheduler`에서 `.forward*(`와 `sample_token_*(`를 찾는 acceptance grep에는 호출이 하나도 걸리지 않는다.
- **attention**: `KVCache`, `RotatingKVCache`(쓰기는 여전히 `update_and_fetch`로 하므로 #2182 undo 로그가 계속 기록된다), `ChunkedKVCache`에 `KvAttention` trait을 구현했다. `attend_batched_rows`가 행별 루프를 공유한다. Gemma 3과 Llama 4가 이를 호출하고, `model_owned.rs`의 분기 헬퍼는 지웠으며, `gemma4_verify_rows`는 `is_dense_fp16()`을 쓴다. `rg 'is_paged_backed\(\)|is_paged_decode\(\)' src/models`에는 아무것도 걸리지 않는다.
- **`DecodeBatchContext` 제거**: core의 `DecodeStorageBackend`도 함께 지웠다. trait 진입점은 `forward_batched_with_ids`가 되었고, VLM wrapper 약 30개와 qwen3_5, qwen3_moe, muse_glimmer의 context 인자 override는 `forward_batched`로 합쳤다.
- **도구**: `mlxcel-engine-parity`에 `d:engine` arm을 추가했다. 제거한 compat 커널을 context로 비교하던 예제와 스크립트는 지웠다.
- **ADR 0009**에 API와 결정을 기록했고, ADR 0007과 0008에는 개정 표시를 달았다.

## 3. 기술적 선택과 그 이유

- **단일 행 처리는 trait이 아니라 엔진에서 한다.** 이슈는 `LanguageModel::forward`를 배치 진입점에 위임하는 provided 메서드로 바꾸자고 했다. 구현이 200개가 넘고 배치 override가 40개쯤이라 모델 트리 전체를 기계적으로 고쳐야 하는 데다, 두 forward 중 무엇을 부를지는 엔진만 정한다. 그래서 엔진을 유일한 decode 진입점으로 두고, B=1에서는 배치 기본 구현이 `b == 1`일 때 하는 것과 같은 단일 행 forward를 부른다. Qwen3과 Llama 3 실제 체크포인트에서 1행 배치 logits가 단일 행 forward와 비트 단위로 같다는 것을 sequence id가 있을 때와 없을 때 모두 테스트로 고정했다.
- **행이 하나인 배치는 항상 행별 체인을 탄다.** 첫 버전은 1행 배치도 fused draw를 타게 했다. fused draw의 readback은 MLX 오류를 돌려줄 수 없어서, B=1에서 예외가 나면 요청 하나만 실패시키지 못하고 프로세스가 중단되었을 것이다(#822). 리뷰에서 이를 잡았고, 이제 fused draw는 B>1에서만 쓰며 여기도 `try_eval`을 거친다.
- **fused 적격성 규칙은 하나만 둔다.** lookahead gate와 동기 step이 같은 `shared_fused_params`를 부른다. 두 규칙이 일치해야만 파이프라인 출력이 동기 경로와 같다.
- **배치 전체의 실패는 한 번으로 센다.** fused draw에서 예외가 한 번 나면 모든 행이 `RowError::BatchEval`로 실패하고 상태 카운터는 한 번만 오른다. 행마다 세면 8행 배치에서 일시적인 예외 한 번으로 `MAX_CONSECUTIVE_EVAL_FAILURES`에 닿아 scheduler가 멈출 수 있다.
- **모델 소유 계열도 캐시를 통해 attend한다.** 이들의 시퀀스별 상태는 pool 기반이 아니므로 ADR 0008의 규칙에서 dense 행만 남는다. 지운 compat 커널은 #2210이 걷어 낸 것과 같은 모양의 행별 C++ 루프였다. 커널 자체는 `ffi_tests` 기준 구현으로 core에 남긴다.
- **burst는 scheduler가 여전히 모델에 직접 닿는다.** 이슈가 burst 생성기를 scheduler에 남기기로 했으므로 MTP와 DFlash burst adapter는 엔진 밖에서 target forward를 부른다. `model()`과 `parts_mut()`는 이 adapter와 handoff 복원을 위해 남았고, 그래서 이 보장은 타입 시스템이 아니라 acceptance grep과 span guard가 지킨다.

## 4. 검증

- clippy(`-D warnings`, 루트는 examples까지, mlxcel-core는 테스트까지), fmt, 라이선스 헤더, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`, `model_loader_stdout_guard`, `prefill_span_coverage`.
- 단위 및 모듈 테스트는 모두 `--test-threads=1`로 하나씩 돌렸다. core는 `engine`, `cache::attend`, `mla::`, decode-undo, paged decode와 detach, sampling, generate, prefill, decode_finish. 루트는 모델 계열(qwen3, llama3, helium, deepseek_v2, gemma3, llama4, gemma4, muse_glimmer, afmoe), `multimodal::`, `server::batch::scheduler` 모듈 전체, active와 sequence 테이블, finish step, speculative burst, disaggregated, pipeline, CLI 입력, engine probe, memory estimate.
- 실제 체크포인트, `main` 병합 후 `MLXCEL_SDPA_DETERMINISTIC=1`:
  - Qwen3-1.7B 4-bit, Llama-3.2-1B 4-bit, Gemma-3-1B 4-bit에서 `mlxcel-engine-parity`: greedy와 seeded-penalties-DRY 모두 CLI, 서버 dense, 서버 paged, 엔진 직접 실행 결과가 같았다. 갈라진 것은 문서화된 prompt-cache 행뿐이고 Phase 0 기준선과 같다.
  - `single_row_batch_parity`: Qwen3과 Llama 3의 1행 배치 logits가 단일 행 forward와 비트 단위로 같다.
  - `scheduler_real_batch_parity`: Gemma 3 greedy 요청 두 개를 동시에 돌렸을 때(decode tick 47개 모두 B=2) 48토큰 출력이 각 요청을 혼자 돌린 결과와 같았다. paged와 dense 백엔드 모두 그랬다.

## 5. 남은 위험

- **처리량은 아직 재지 않았다.** decode tick마다 작은 Vec 할당 두 개(행 view와 결과)가 남아 있다. 1.0% 허용치는 epic 마지막 측정에서 확인한다.
- **Llama 4는 로컬 체크포인트가 없다.** 이제 dense 백엔드에서도 RoPE 층 블록을 배치로 돌리는 배치 경로는 합성 테스트로만 확인했다.
- **비트 단위로 같지 않은 연산 순서 변화**: paged 백엔드의 Gemma 3과 Llama 4는 compat 커널 대신 행별 `attend`를 쓴다. dequant-first나 압축 변형이 있는 Turbo 모드의 Gemma 3 global 층은 그 변형을 쓴다. Llama 4 RoPE 층은 dense 백엔드에서도 배치로 돈다.
- **Gemma 3과 Llama 4는 여전히 `supports_paged_decode_backend() == true`를 반환한다.** 그래서 `auto`가 이들에게 paged를 고르고, pool 할당과 step마다의 동기화 비용을 치르는데 attention은 pool을 읽지 않는다. 측정해서 정할 후속 과제로 남겼다.
- **pipelined 경로와 logprobs의 readback**은 main과 마찬가지로 오류를 돌려주지 못한다.

## 6. 학습 포인트

- **문서는 의도를 적고, 동작을 고정하는 것은 테스트다.** 코드 주석, 커밋 본문, ADR이 모두 '1행 배치는 행별 체인을 탄다'고 적었지만 코드는 그렇지 않았다. B=1이 어느 경로를 탔는지 확인하는 테스트 하나가 있었다면 바로 드러났을 것이다.
- **실패는 실패한 연산의 단위로 센다.** 여덟 행을 덮는 eval 하나는 실패 한 번이고 여덟 번이 아니다. 상태 카운터의 한도도 한 번을 기준으로 정해져 있었다.

## 7. 관련 항목

- epic #2166, 이슈 #2172, ADR 0007, 0008, 0009.
- #2168(finish step), #2169(row sampler), #2170(prefill plan), #2171 / #2210(캐시가 고르는 attention), #822(행 단위 오류), #2182(decode-undo 로그), #2173과 #2176(다음 단계).
