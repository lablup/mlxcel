# 기술 보고서: PR #2201 - CLI와 서버의 모든 decode 경로에 샘플링 후 종료 처리 하나

**작성일**: 2026-10-07

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust

**위험도**: 중간. 이제 모든 decode 경로가 한 함수를 거쳐 종료한다. 실제 체크포인트 두 개에서 greedy 출력은 변경 전후 바이트 단위로 같다. 순서나 주기가 바뀐 곳 네 군데가 남아 있고 모두 PR 본문에 밝혀 두었는데, 기본 설정에서는 실행되지 않거나 출력에 영향이 없다.

## 요약

epic #2166의 Phase 1(#2168)이다. 토큰을 샘플링한 뒤 각 decode 경로는 시퀀스를 끝낼지 판단한다. 판단 항목은 EOS, 이력 push, stop 문자열, 생성 한도, 구조화 출력 stop, `max_tokens`, 컨텍스트 한도, 루프 감지, 캐시 정리 주기다. 이 판단이 `CxxGenerator` 루프 네 곳과 서버 경로 다섯 곳, 모두 아홉 군데에 따로 쓰여 있었다. 이제는 `mlxcel_core::decode_finish::finish_step` 하나뿐이다. 서버는 `src/server/batch/finish.rs`의 hooks 타입 하나를 통해 이 함수를 호출하고, `FinishCause`를 `FinishReason`으로 바꾸는 대응도 그곳 한 군데에만 있다. 그 결과 speculative burst 경로(MTP, DFlash)에도 루프 감지와 컨텍스트 한도가 적용된다.

## 1. 문제 정의

종료 처리 코드는 복사본마다 조금씩 달라져 있었다.
- stop 문자열 지원을 넣을 때 출력 지점 다섯 곳과 마무리 지점 두 곳을 고쳐야 했다.
- 생성 한도는 "세 경로 모두"에 넣어야 했다.
- speculative burst 경로에는 루프 감지가 아예 없었다.
- 한 곳을 고쳐도 나머지 복사본에는 반영되지 않았다. epic이 없애려는 버그가 바로 이런 종류다.

## 2. 변경 요약

- **`decode_finish`**: `finish_step(input, hooks) -> Option<FinishCause>`가 정해진 순서 하나로 검사한다.
  1. EOS(push하지 않음)
  2. push와 이력 기록
  3. stop 문자열
  4. 생성 한도
  5. 구조화 출력 stop
  6. `max_tokens`
  7. 컨텍스트 한도
  8. 루프 감지

  캐시 정리는 종료하지 않았을 때만 돈다. CLI는 `NoStopHooks`를 쓴다.
- **`server::batch::finish`**: 서버용 `FinishHooks` 구현과 종료 원인 대응 하나. 배치 행 단위, fused, 단일 step, prefill 완료, burst 경로가 이것을 쓴다.
- **burst 경로**: 토큰마다 같은 함수로 종료를 판단한다. 루프 감지와 컨텍스트 한도가 새로 적용되고(`ContextBound`를 `BurstContext`로 전달), 일치한 stop의 우선순위와 디토크나이저 flush는 그대로 유지한다.
- **burst의 캐시 저장 차단**(보안 검토에서 나옴): 생성기가 이미 만든 배치의 마지막 토큰보다 앞에서 끝난 burst는 이유와 상관없이 상태를 prompt cache에 넣지 않는다. 남은 토큰은 모델 상태에는 이미 반영됐지만 `generated_tokens`에는 없어서, snapshot을 재사용하는 모델 계열이라면 캐시 키보다 앞선 상태를 저장하게 되기 때문이다. 이 수정으로, 이전부터 있던 stop 문자열과 생성 한도 쪽의 같은 빈틈도 함께 막혔다.
- **삭제**: `finish_on_generation_bound`, `context_bound_stop_due`. 같은 규칙은 `ContextBound::stop_due`가 그대로 지킨다.

## 3. 기술적 선택과 그 이유

**양쪽이 공유하는 고정된 순서 하나.** 이슈가 순서를 하나로 정하라고 요구하므로, 경계 사례 몇 가지가 이제 경로마다 따로가 아니라 모든 곳에서 같게 처리된다. 바뀐 점은 PR 본문에 밝혔다.
- prefill에서 첫 토큰이 구조화 출력 stop과 생성 한도에 동시에 걸리면 `length`로 보고한다.
- CLI에서 `max_tokens`를 다 쓰는 동시에 루프를 완성하는 토큰은 출력되고 `Length`로 보고된다. CLI는 기본적으로 루프 감지를 켜지 않으므로 기본 설정에서는 일어나지 않는다.
- 캐시 정리 주기의 기준이 생성 길이로 바뀌어 한 토큰 밀렸고, burst와 prefill 경로에서도 돈다. CUDA에서는 캐시 정리가 기본적으로 꺼져 있다.

**`generated_len`을 hooks에 넘긴다.** `finish_step`이 `generated_tokens`를 가변으로 빌리고 있는 동안 서버 hook은 `SequenceInfo`의 다른 필드를 따로 빌리므로, 길이를 스스로 읽을 수 없다. 길이를 인자로 넘기면 토큰마다 복제나 할당을 하지 않아도 된다.

## 4. 검증

- 단위 테스트
  - `decode_finish` 9, `generate::finish_tests` 6, `generate` 61, `loop_detection` 22.
  - 서버 모듈은 하나씩 돌렸다: `finish_step_tests` 13(같은 시나리오를 경로 종류마다 한 번씩, 그리고 캐시 저장 차단), `speculative_burst_tests` 58, `speculative_slice` 25, `stop_sequence_tests` 8, `scheduler::tests` 95, `sequence` 30, `scheduler_prompt_cache_tests` 31.
  - burst 루프 테스트는 `main`에서 실패한다.
- GB10 실제 체크포인트, `mlxcel-engine-parity`를 `MLXCEL_SDPA_DETERMINISTIC=1 --no-prompt-cache-case --expect-identical`로 실행
  - 모델: qwen3-1.7b-4bit, llama-3.2-1b-instruct-4bit
  - 실행: `max_tokens`에서 끝나는 256토큰 생성, 한 단어 프롬프트로 EOS에서 끝나는 생성
  - CLI, 서버 dense, 서버 paged의 출력이 24쌍 모두 같았다.
  - 모든 토큰 열이 변경 전 바이너리의 결과와 바이트 단위로 같았다.
- 처리량은 여기서 재지 않았다. epic이 마지막에 ADR 0007의 1.0% 허용치를 기준으로 한 번에 잰다.

## 5. 남은 위험

- 요청에서 지정하는 루프 감지 `min_count`에 상한이 없다(이전부터, LOW).
- burst의 thinking budget 덮어쓰기는 바뀌지 않았다(이전부터).
- `prompt_lookup.rs`는 여전히 `detect_repetition_loop`를 직접 호출한다. 아홉 경로에 들어가지 않아서 Phase 6으로 넘겼다.
- 스케줄러 테스트 모듈 두 개를 한 프로세스에서 돌리다 `cudaStreamEndCapture` 중단이 한 번 났다. 이후 함께 돌린 실행은 통과했고, `main`과는 비교하지 않았다.

## 6. 학습 포인트

- **복사본을 하나로 합치면 숨어 있던 차이가 드러난다.** PR 본문에 밝힌 동작 변경 하나하나가 원래 복사본끼리 이미 서로 다르게 동작하던 지점이다.
- **새 종료 경로는 상태 저장의 새 위험을 만들 수 있다.** 배치 중간 종료는 텍스트 출력에는 안전했지만 snapshot 저장에는 안전하지 않았다. 이 변경으로 배치 중간 종료가 잦아졌기 때문에 보안 검토에서 이 문제가 드러났다.

## 7. 관련 항목

- epic #2166, 이슈 #2168.
- ADR 0007.
- 함께 진행된 #2169, #2170.
