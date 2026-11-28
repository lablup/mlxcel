# PR #3330: K2-Horizon 예산 종료 태그를 열린 변형에 맞추기

**날짜**: 2026-11-23
**상태**: PR 열림 (#3321 종료). 단위 테스트 통과, M3 Ultra에서 `mlxcel-server`로 `models/k2-horizon-0.9b` 실측 확인

## 요약

IFM K2-Horizon의 채팅 템플릿은 `reasoning_effort`에 따라 세 가지 태그 중 하나로 추론 블록을 엽니다: `<ifm|think>` (high), `<ifm|think_fast>` (medium), `<ifm|think_faster>` (low). 각 태그에는 짝이 되는 닫는 태그가 따로 있습니다. #3303에서 thinking-budget 추적기가 세 종류의 닫는 태그를 모두 인식하게 되었지만, `thinking_budget_tokens` 상한에 도달하거나 `reasoning_end` 제어 요청이 들어오면 항상 `</ifm|think>`를 강제했습니다. 그래서 `medium`이나 `low` 요청에서는 모델이 학습 중에 한 번도 본 적 없는 열기/닫기 조합이 만들어졌습니다. 또한 모델이 스스로 출력한 변형 여는 태그는 인식되지 않아, 그런 블록에는 예산이 적용되지 않았습니다.

이 PR은 `ThinkingState`가 블록을 연 태그(프롬프트가 미리 연 태그 또는 모델이 직접 출력한 태그)에 짝이 되는 닫는 태그를 기억하게 하고, 두 강제 경로 모두 그 태그를 내보내게 합니다. 실제 체크포인트에서 예산 6인 `medium` 요청은 이제 여섯 토큰 뒤에 `</ifm|think_fast>`로 추론을 끝냅니다. 또한 #2442 이후 더 이상 발생하지 않는 panic을 기대하던 오래된 Qwen3.5 ignored 테스트를 교체합니다.

## 1. 문제

`resolve_thinking_token_ids`는 `close = </ifm|think>`인 `ThinkingTokenIds { open, close, alt_closes }`를 반환합니다. `ThinkingState::decide_override`는 두 강제 경로 모두에서 `ids.close`를 강제했습니다. 하나는 프롬프트가 블록을 연 Pending 경로(예산 0이거나 첫 토큰 전에 force-end가 걸린 경우)이고, 다른 하나는 InBlock 경로(상한 도달 또는 force-end)입니다. `observe`는 `ids.open`에서만 블록에 진입했습니다.

| 경우 | 이전 | 이후 |
|---|---|---|
| `medium` 프라이밍, 상한 도달 | `</ifm|think>` | `</ifm|think_fast>` |
| `low` 프라이밍, 상한 도달 | `</ifm|think>` | `</ifm|think_faster>` |
| 프라이밍 없이 모델이 `<ifm|think_fast>` 출력 | Pending에 머물고 예산 미적용 | InBlock 진입, `</ifm|think_fast>`로 종료 |
| Qwen, Gemma 4, `<|content_thinking|>` | 쌍의 닫는 태그 | 변화 없음 |

## 2. 변경 사항

| 영역 | 변경 |
|---|---|
| `server/thinking_budget.rs` | `ThinkingTokenIds::alt_opens`를 추가했습니다. `alt_closes`와 인덱스로 짝을 이루며 같은 `THINK_EFFORT_VARIANTS` 항목에서 채웁니다. `close_for_open`도 추가했습니다. `ThinkingState`에는 비공개 필드 `active_close`가 생겼습니다. `observe`가 여는 태그를 볼 때, 그리고 프롬프트가 블록을 연 경우 `with_primed_close`가 이 값을 설정하며, `ids.is_close(id)`일 때만 받아들입니다. 두 강제 경로는 `forced_close()`(`active_close.unwrap_or(ids.close)`)를 호출합니다. 새 함수 `resolve_primed_close_id`는 마커 문자열을 토큰 id로 바꿉니다. |
| `server/config.rs` | `ServerGenerateOptions::thinking_primed_close: Option<String>` |
| 라우트 | 프롬프트 프라이밍 여부로 `thinking_enter_block_on_start`를 설정하는 모든 곳에서 `primed_open_thinking_close_marker`로 새 필드도 함께 설정합니다. 대상은 `prepare_chat_generation`(chat completions, 그리고 `InProcessServer::chat`을 거치는 `mlxcel chat` / `mlxcel run` CLI), `/v1/responses`의 run 경로와 non-streaming 경로, responses compaction, Anthropic `/v1/messages`(두 경로), ASR 채팅 스트림, router front입니다. |
| `batch/scheduler` | `build_thinking_state`가 마커를 받아 `resolve_primed_close_id`로 변환하고 `with_primed_close`로 붙입니다. |
| `models/qwen3_5_tests.rs` | `rollback_speculative_cache_panics_on_ragged_batched_accepts`를 `rollback_speculative_cache_compacts_ragged_batched_accepts`로 교체했습니다. |
| 구조체 리터럴 | warmup, engine probe, request-options 빌더, 테스트 리터럴에 `thinking_primed_close: None`을 추가했습니다. |
| 문서 | `docs/supported-models.md`(K2-Horizon 항목), `CHANGELOG.md`(Fixed) |

## 3. 기술적 결정

### 3.1 닫는 태그는 id 테이블이 아니라 상태에 둔다

id 테이블은 토크나이저 단위로 만들어져 모든 시퀀스가 공유합니다. 반면 어떤 블록이 어떤 변형으로 열렸는지는 시퀀스마다 다릅니다. `ThinkingState`에 `Option<i32>` 하나를 두면 핫패스 비용이 사실상 없습니다. `forced_close`는 강제가 필요할 때만 실행되고, `close_for_open`은 프라이밍이 없는 Pending 상태에서만 실행됩니다(비교 한 번과 두 칸 스캔). `ThinkingTokenIds`는 계속 `Copy`입니다.

### 3.2 프라이밍된 닫는 태그는 라우트가 이미 계산한 문자열로 admission 시점에 변환한다

라우트는 스트림 필터와 #1470 delimiter echo를 위해 이미 `primed_open_thinking_close_marker`를 호출합니다. 그 문자열을 `ServerGenerateOptions`에 실어 보내고, admission에서 스케줄러의 토크나이저로 한 번만 변환하면 라우트가 토큰 id를 다룰 필요가 없습니다. `enter_block_on_start`가 참일 때만 변환하고, `with_primed_close`는 쌍의 닫는 태그가 아닌 id를 버립니다. 그래서 여러 토큰으로 된 닫는 태그(`<|close|>think<|sep|>`)나 다른 계열의 마커가 들어오면 엉뚱한 토큰을 강제하지 않고 쌍의 닫는 태그로 대체됩니다.

### 3.3 닫는 태그가 없는 변형 여는 태그는 블록을 열지 않는다

어휘에 닫는 태그가 없는 변형에 대해서는 `close_for_open`이 `None`을 반환합니다. 이때 `ids.close`로 대체하면 이 PR이 없앤 잘못된 짝이 다시 생깁니다. 실제 K2 어휘에는 여섯 토큰이 모두 있으므로 이론적인 경우이며, 이유는 doc comment에 적어 두었습니다.

### 3.4 분산(disaggregated) handoff는 범위 밖

router front에서도 이 필드를 설정하지만, prefill 프레임의 샘플링 페이로드(`kv_cache_serde`)가 이 필드를 전달하지 않습니다. 그래서 프라이밍된 변형에 대해 decode 노드의 `build_handoff_thinking_state`는 여전히 `</ifm|think>`를 강제합니다. 와이어 포맷 변경은 #3321에서 명시적으로 제외되었습니다. handoff가 생성된 토큰을 `observe`로 다시 재생하므로, decode 쪽에서 모델이 직접 출력한 변형 여는 태그는 이제 추적됩니다. router front 대입 위치의 주석과 문서에 이 한계를 적었습니다.

### 3.5 낡은 것은 코드가 아니라 Qwen3.5 테스트의 계약이었다

#2442 이후 `rollback_speculative_cache_rows`는 행마다 수락 개수가 다른(ragged) 경우에 panic하지 않습니다. 대신 각 행의 수락된 블록을 압축하고 행별 gated-delta 상태를 다시 계산합니다. 새 테스트는 `[1, 2, 3, 4]`와 `[5, 6, 7, 0]` 두 행을 수락 개수 `[3, 1]`로 검증합니다. 각 행의 `conv_state`와 `state_cache`가 그 행만 자기 수락 개수로 B = 1 롤백한 결과와 같은지, 그리고 attention offset이 `max(accepted) + 1` = 4인지 확인합니다. 공용 헬퍼는 반환값을 `max(accepted)`와 비교하도록 바꿨습니다. 함수 문서에 따르면 uniform과 ragged 모두 이 값을 반환합니다.

## 4. 검증

| 셀렉터 | 결과 |
|---|---|
| `server::thinking_budget server::k2_horizon_reasoning_tests` | 통과 (thinking_budget 기존 테스트는 변경 없음, K2 테스트 12개 중 신규 6개) |
| `server::routes::chat` | 통과 |
| `models::qwen3_5_tests --ignored --test-threads=1` (GPU lock) | 7개 모두 통과 |
| `coverage_tests` | 18개 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --check`, CI 계약 스크립트 7개 | 이상 없음 |

실제 체크포인트 검증 조건: `models/k2-horizon-0.9b`에서 `mlxcel-server --skip-chat-parsing`, greedy, `thinking_budget_tokens: 6`, 프롬프트 "What is 17 times 23? Think step by step."

| effort | 원시 출력 앞부분 |
|---|---|
| high | `The user asks: "What</ifm|think>\nWe need to answer: ...` |
| medium | `We need to compute 17</ifm|think_fast>\nWe need to answer: ...` |
| low | `</ifm|think_faster>\nLet's calculate ...` (모델이 첫 토큰에서 스스로 닫음) |

파싱을 켜면 세 effort 모두 예산 안의 추론 토큰 여섯 개가 `reasoning_content`로, 나머지가 `content`로 들어갑니다. 예산이 없으면 `medium`은 39번째 토큰에서 `</ifm|think_fast>`로 스스로 닫습니다.

## 5. 알려진 한계

- 분산 prefill/decode handoff는 프라이밍된 변형에 대해 여전히 `</ifm|think>`를 강제합니다 (3.4절).
- 강제된 토큰은 chat `logprobs` 배열에 나타나지 않습니다. 검증 중에 관찰했으며, 이 변경과 무관한 기존 동작입니다.
