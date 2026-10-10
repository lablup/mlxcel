# 기술 보고서: PR #2262 - 일반 디코드를 느리게 하지 않는 서버 prompt lookup

**작성일**: 2026-10-11

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다. 속도 기준과 임시 상수 두 개는 실행 마지막 측정으로 정한다.

**언어**: Rust, Python(벤치마크 클라이언트), Markdown(ADR, 문서, CHANGELOG)

**위험도**: 중간. 기능은 선택형(`--draft-kind prompt-lookup`)이지만 batch scheduler의 lookahead gate를 바꾸고, pipelined tick 옆에 두 번째 tick 형태(동기 verify tick)를 추가한다. 실제 체크포인트에서 출력은 정확한 logit 동점을 빼면 플래그를 끈 서버와 같다. lookahead와 append 소유 규칙은 불변식으로 문서화하고 테스트로 고정했다.

## 요약

prompt-lookup 디코딩은 `mlxcel generate --prompt-lookup`에서만 돌았다. 서버는 `--draft-kind prompt-lookup`을 거부했고, speculative dispatch를 켜면 모든 토큰에서 lookahead pipeline이 꺼져 일반 디코드가 약 12퍼센트 느려질 구조였다. 이 PR은 `mlxcel serve`와 `mlxcel-server`가 batch scheduler 안에서 시퀀스별 prompt-lookup verify round를 돌리게 한다. round는 tick당 행마다 최대 한 번이고, 어느 행도 제안하지 않는 tick은 플래그가 없을 때와 똑같이 pipeline으로 돈다. 배치 크기 gate(`--prompt-lookup-max-batch`, 임시 기본값 2)와 drafter의 governor가 행의 verify 여부를 정하므로 verify forward가 배치 디코드를 밀어내지 않는다. verify round 자체는 CLI 클라이언트에서 `Engine::verify_round`로 옮겨, CLI와 서버가 같은 코드를 쓴다.

## 1. 문제 정의

- `SpeculativeDispatch::resolve`는 `--draft-model`이 없으면 `--draft-kind`를 읽기 전에 `Disabled`를 반환했고, draft model이 있으면 `prompt-lookup`을 알 수 없는 값으로 거부했다.
- `BatchScheduler::lookahead_params`는 `should_dispatch_speculative()`가 참이면 항상 `None`을 반환했다. 단순히 prompt-lookup arm을 붙였다면 모든 일반 토큰이 동기로 돌았을 것이다.
- 제안하는 행마다 별도의 다중 토큰 verify forward가 필요하다. 동시 요청이 많으면 배치 step 위에 제안 행마다 forward가 하나씩 더해져 전체 처리량이 떨어질 수 있다.
- paged 저장소에서 다중 토큰 verify forward는 pool gather 경로를 타며, ADR 0001은 4096 토큰 이상에서 이 경로를 contiguous SDPA의 2~3배 비용으로 측정했다.
- 메인테이너는 이 기능 제공이 일반 디코드를 느리게 하면 안 된다고 요청했고, 그래서 이슈에 정확성 기준과 함께 속도 기준이 들어 있다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `server/speculative_dispatch.rs` | `SpeculativeDispatch::PromptLookup { config, max_batch }`, draft model 조기 반환 전에 해석. draft model과 함께 쓰면 `InvalidKind`. `--draft-block-size`가 `max_draft`를 정한다. `is_kind_specific()`은 거짓으로 남아 burst 경로에 들어가지 않고, scheduler는 `prompt_lookup_config()`를 읽는다. `--spec-type none`도 이를 끈다. `summary()`는 `speculative=prompt-lookup (max_draft=N, policy=P, max_batch=M)`을 출력한다. |
| 서버 CLI | `mlxcel serve`와 `mlxcel-server`에 `--prompt-lookup-max-batch N`(기본값 `DEFAULT_PROMPT_LOOKUP_MAX_BATCH = 2`, 임시). |
| `engine/verify_round.rs`(신규) | `Engine::verify_round`(verify forward, 모든 위치를 순서대로 per-row 샘플링, 수락된 접두부와 target 토큰 commit, 나머지 unwind), `Engine::verify_rounds_unsupported`(하나의 적격 규칙: 토큰 전용 sampler, trim 가능 상태, 모델 지원), `Engine::warm_up_verify_widths`. `DirectEngine::speculate`가 이들을 호출하며, `engine/speculative.rs`에서 약 150줄이 빠졌다. |
| `server/batch/scheduler/prompt_lookup.rs`(신규, 615줄) | `SequenceInfo`에 행별 drafter, prefill 완료 시 준비. tick별 질의, verify gate, pipelined와 동기 tick 사이 전환, 첫 적격 prefill에서 verify 폭 예열. 모듈 문서에 불변식 I1~I8(동기 상태, in-flight append 하나, 제안 뒤 prime 금지, drafter 문맥, gate, 구성원 변화, 오류, 취소)을 적었다. |
| `decode_tick.rs`, `prefill.rs`, `finish.rs` | `lookahead_params`는 prompt lookup에 대해서만 일괄 speculative 검사를 뺀다(MTP와 DFlash는 유지). `ContextBound::verify_room`이 제안 길이를 제한해 verify forward가 컨텍스트 한도 정지 위치를 넘어 append하지 않는다. |
| `prompt_lookup_drafter.rs` | 일반 step 토큰을 위한 `observe_emitted`. n-gram 인덱스를 prefill이 아닌 첫 조회 때 만든다. |
| `scripts/bench_serving_concurrency.py` | `--prompt-style plain|copy`. |
| 문서 | ADR 0009(#2255로 개정), `docs/CONTINUOUS_BATCHING.md`, `docs/llama-server-compat.md`, `docs/speculative-acceptance.md`, CHANGELOG Unreleased. |
| 테스트 | `speculative_dispatch_tests`(신규 5), `engine::verify_round_tests`(3), `scheduler_prompt_lookup_tests`(11). 마지막은 가중치 0 레이어로 greedy 다음 토큰이 `(t + 1) % 16`이 되는 dense Llama 위에서 돌고, tick마다 각 행의 KV offset이 commit된 토큰 수와 같은지 확인한다. |

파일 48개, +2433 / -141.

## 3. 기술적 선택과 그 이유

**burst가 아니라 tick당 verify round 하나.** 끝까지 도는 burst(DFlash arm 형태)는 요청 내내 worker를 붙잡아 다른 모든 행을 멈추게 한다. #734가 MTP에서 없앤 head-of-line block이다. tick당 round 하나면 다른 행도 매 tick 토큰을 받고, round 상태(drafter, 카운터)는 tick 사이에 `SequenceInfo`에 남는다.

**lookahead 전환 규칙.** scheduler는 step n+1 prime을 제출하기 전에 gate를 통과한 모든 drafter에 제안을 묻는다. 아무도 제안하지 않으면 그 tick은 바뀌지 않은 pipelined tick이다. 누군가 제안하면 step n을 `try_eval`로 읽어 prime 없이 commit하고, 질의를 retract해 governor가 가짜 miss를 기록하지 않게 하며, 다음 tick을 동기로 돌린다. 이때 제안 행은 verify하고 나머지는 동기 배치 step을 탄다. 다시 prime하는 것은 아무도 제안하지 않고 gate를 통과한 모든 drafter가 `pipelines_plain_rounds()`를 보고할 때뿐이다. CLI의 `pipelined_plain_round`와 같은 방식이라 전환에 forward 비용이 들지 않는다. 그 결과, 새 요청이 들어오면 그 요청의 drafter가 세 번 연속 제안하지 않을 때까지 모든 행이 동기로 남는다. 이슈 본문이 요구한 규칙이며, 비용은 측정 (a)와 (c)로 확인한다.

**verify gate.** 행은 governor가 예산을 주고 활성 디코드 배치가 `--prompt-lookup-max-batch` 이하일 때만 verify한다. 임시 기본값 2의 근거는 B <= 2에서 제안 행마다 forward가 하나 더해져도 step 비용이 최대 두 배라는 점이다. gate가 거절한 행은 `draft_block`을 부르지 않아 governor가 miss를 기록하지 않고, commit된 토큰은 계속 관찰해 배치가 줄면 다시 참여한다.

**paged 한도는 미정.** `PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT`은 측정 (d)가 8192 토큰에서 gather 경로 verify 비용이 이득을 상쇄하는지 보여 줄 때까지 `None`(한도 없음)이다.

**리뷰와 보안 수정.** 리뷰는 서버 쪽 verify 폭 예열을 추가했고(CLI는 예열했지만 서버는 하지 않아 초기 요청이 디코드 도중 kernel 첫 사용 비용을 냈다), n-gram 인덱스 생성을 첫 조회로 옮겼다(gate가 거절한 행이 prefill에서 비용을 내지 않는다). 보안 리뷰는 제안 길이를 컨텍스트 한도로 제한했고(`--max-kv-size` 바로 아래에서 끝나는 프롬프트가 정지 뒤 버려질 위치를 최대 63개까지 append하게 할 수 있었다), 같은 한도를 예열에도 적용했으며, 질의 경로의 경계 검사 없는 slice를 바꿨다. scheduler에는 `catch_unwind`가 없어 거기서 panic이 나면 서비스 전체가 멈추기 때문이다.

## 4. 검증

- 최종 브랜치의 로컬 게이트(GB10, `--profile test-fast --features cuda`, `gpu-lock`, `--test-threads=1`): fmt, clippy `-D warnings`(루트와 mlxcel-core), mlxcel-core `engine::`, `speculative::`, `drafter::`, `session`, `sampling`, `generate::`, 루트 `server::batch`, `server::speculative_dispatch`, `server::in_process`, `engine_probe`, `backend::`, `cli_input`, `commands::`, `mlxcel-server` bin 테스트, `cli_help_consistency`, `dead_doc_pointers`, `model_loader_stdout_guard`, Apache 헤더 검사. 모두 통과했다. `make verify-test-cuda`(workspace, 13,868 통과, 398 무시)는 이 PR이 만든 회귀 하나를 찾았다. 새 `prompt_lookup_max_batch` 필드를 runtime settings schema에 분류하지 않아 `server_config_schema_classifies_all_111_fields`가 실패했다. 9f5b5a87에서 고쳤다(다른 speculative 설정처럼 읽기 전용, 정수, worker 범위). 나머지 두 실패는 #2230에서 다루는 기존 SSM bf16 parity 테스트다.
- Llama-3.2-1B `mlxcel generate --prompt-lookup`, greedy, 160 토큰: 이 PR 이전 빌드와 출력이 같다.
- 서버 쌍, release 빌드, `MLXCEL_SDPA_DETERMINISTIC=1`, `/completion` greedy, 플래그 켬과 끔: Qwen3-1.7B 140 토큰(auto, dense, paged 저장소)과 Llama-3.2-1B 160 토큰(auto, paged)에서 복사 응답과 일반 응답이 같고, 플래그를 켠 서버에 복사와 일반 요청을 동시에 보내도 각각 단독 결과와 같으며, `draft_kind`는 `prompt-lookup`, `draft_n_accepted > 0`이다. `mlxcel serve`도 같다. Gemma-3-1B는 prompt lookup이 동작하지 않은 채 같은 응답을 낸다. sliding-window `RotatingKVCache`는 trim할 수 없어 공유 적격 규칙이 거절하며, CLI에서도 마찬가지다.
- Qwen3 복사 응답은 160 토큰에서 복사가 끝난 뒤 약 150번째 토큰에서 한 번 달라진다. 플래그를 끈 서버의 그 위치 상위 두 후보 logprob가 -1.625로 같은 정확한 동점이다.
- `mlxcel-engine-parity -n 400`, Qwen3-1.7B와 Llama-3.2-1B: #2256과 같다.

## 5. 남은 위험

- 속도는 아직 측정하지 않았다. 실행 마지막 측정은 (a) 일반 질문 B=1에서 플래그 켬과 끔의 차이 -1.0퍼센트 이내, (b) 복사가 많은 B=1의 속도 향상과 수락 수, (c) 동시성 2, 4, 8에서 플래그 끔, 켬, gate를 연 켬 비교(`DEFAULT_PROMPT_LOOKUP_MAX_BATCH` 결정), (d) paged와 dense 저장소의 8192 토큰 프롬프트(`PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT` 결정)를 다룬다.
- 플래그가 켜져 있고 배치가 gate 안에 있으면, 요청이 들어올 때마다 약 세 tick 동안 모든 행이 동기로 돈다.
- n-gram 인덱스는 프롬프트 토큰당 약 120바이트이고 첫 조회 때 scheduler 스레드에서 만든다(128K 토큰에서 약 15MB, 수십 ms).
- verify eval 실패를 주입하는 scheduler 수준 테스트는 없다. 엔진 수준 실패 경로는 `verify_round_tests.rs`가 다룬다.
- 규칙 3이 제안하지 않은 행의 빈 질의도 retract하므로 `stats.rounds`가 적게 집계된다(CLI도 같고, 클라이언트에는 보이지 않는다).

## 6. 학습 포인트

- speculative 디코딩과 lookahead pipelining은 배타적이지 않다. prime 전에 drafter에 묻고, 제안이 있으면 prime 없이 commit하면, 제안 없는 tick은 forward 비용 없이 완전히 pipeline을 유지한다.
- 전환 로직보다 append 소유 불변식(각 시점에 어떤 미commit 위치를 누가 들고 있고, 누가 commit하거나 unwind하는지)을 먼저 적으니 테스트마다 불변식 하나를 겨눌 수 있었고, tick마다 KV offset과 commit 토큰 수를 비교하는 검사가 누수를 잡는다.
- 한 문장을 반복해 만든 합성 벤치마크 프롬프트는 prompt lookup 입장에서 복사가 많은 프롬프트다. 손실 없음 측정에는 반복이 없는 프롬프트가 필요하다.

## 7. 관련 자료

- 이슈 #2255(#2229에서 분리), epic #2166, PR #2256(#2229), PR #2225(`DirectEngine`, `PromptLookupDrafter`), ADR 0001, ADR 0009, #734, #822.
