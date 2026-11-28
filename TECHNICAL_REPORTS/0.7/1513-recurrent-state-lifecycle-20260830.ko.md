# 기술 보고서: PR #1513 - recurrent-state-lifecycle

**작성일**: 2026-08-30

**상태**: Synthetic validation 한계가 있는 구현 완료

**언어**: Rust

**위험도**: High

## 요약

PR #1513은 issue #1220을 해결하기 위해 RWKV7, RecurrentGemma, KimiLinear를 기존 model-owned `SequenceState` lifecycle에 연결한다. 기존 구현은 각 `LanguageModel::forward()` 호출 안에서 recurrent 또는 mixed recurrent-attention cache를 새로 만들었기 때문에, prompt prefill 이후 한 token씩 호출되는 일반 greedy decode loop에서 이전 hidden history가 사라졌다.

이번 변경은 세 family를 계속 non-batched로 유지하고, server scheduler 경로에서는 per-`SequenceId` model-owned state를 사용한다. CLI와 benchmark 같은 single-stream 경로는 fallback slot을 사용하되, 새 multi-token prefill 직전에만 reset한다. Scheduler가 `SequenceId`를 넘겼는데 준비된 state slot이 없으면 fallback state를 조용히 쓰지 않고 fail closed한다. EOS id도 hardcoded family constant가 아니라 `read_eos_token_ids()`를 통해 checkpoint sidecar/config에서 읽는다.

이 worker에서 수행한 검증은 deterministic/synthetic 범위다. Rust compilation, state lifecycle behavior, EOS metadata parsing, formatting, whitespace check를 확인했다. 그러나 pinned checkpoint와 hardware target이 제공되지 않았으므로 실제 RWKV7, RecurrentGemma, KimiLinear checkpoint generation은 검증하지 않았다.

## 1. 문제 정의

영향을 받는 model family는 recurrent 또는 hybrid recurrent-attention architecture다.

- RWKV7은 layer별 token-shift, recurrent state, FFN cache를 저장한다.
- RecurrentGemma는 layer별 RGLRU recurrent state와 rotating local-attention cache를 저장한다.
- KimiLinear는 layer별 MLA attention cache와 GatedDeltaNet convolution/SSM state를 저장한다.

Generation contract는 여러 번의 호출로 구성된다. 보통 prompt prefill을 먼저 실행하고, 이후 greedy decode step마다 token 하나로 model을 다시 호출한다. 이때 forward call 안에서 recurrent state를 매번 새로 만들면 decode step은 prefill history나 이전 generated token을 볼 수 없다. 이는 성능 문제가 아니라 correctness bug다.

## 2. 변경 요약

| 영역 | 변경 |
|------|------|
| RWKV7 | Per-forward local cache allocation을 `ModelOwnedSequenceState<Rwkv7Cache>`로 교체하고, prepare/release/forward-by-sequence hook, fallback reset, model-owned layout, EOS metadata 저장을 추가했다. |
| RecurrentGemma | Mixed recurrent/attention cache를 위한 `ModelOwnedSequenceState<GriffinLayerCache>`를 추가하고 scheduler lifecycle hook, fallback reset, model-owned layout, EOS metadata 저장을 연결했다. |
| KimiLinear | Mixed MLA/Delta cache를 위한 `ModelOwnedSequenceState<KimiLinearCache>`를 추가하고 scheduler lifecycle hook, fallback reset, model-owned layout, EOS metadata 저장을 연결했다. |
| Shared state helper | Scheduler path가 prepared state 누락을 fallback state로 숨기지 않도록 `with_existing_sequence_state()`를 추가했다. |
| EOS loading | `read_eos_token_ids()`가 `generation_config.json`, `tokenizer_config.json`, `config.json` 순서로 EOS를 읽도록 확장했다. Tokenizer의 `eos_token` string은 `added_tokens_decoder`로 id를 찾는다. |
| Tests | EOS parsing, sequence-state continuity, per-sequence isolation, release, missing-state rejection에 대한 focused deterministic coverage를 추가했다. |

### 통계

| 항목 | 값 |
|------|----|
| 구현 변경 파일 수 | 9 |
| 구현 diff | +819 / -102 |
| 주요 구현 커밋 | `b1836a2a` `fix(models): persist recurrent decode state` |
| PR | #1513 |
| Issue | #1220 |

## 3. 기술적 선택

### 3.1 External KVCache state 대신 model-owned state 사용

`LanguageModel` trait는 homogeneous `KVCache` slice를 제공하지만, 이 family들의 state는 heterogeneous recurrent state다. RWKV7은 RWKV-specific layer state, RecurrentGemma는 RGLRU와 rotating KV cache, KimiLinear는 MLA와 Delta cache를 섞어 쓴다. 기존 Mamba, Mamba2, Jamba, Plamo2, FalconH1, NemotronH, Qwen3-Next와 같은 pattern을 사용하면 incompatible external KV slot에 억지로 맞추지 않아도 된다.

### 3.2 CLI path용 fallback state는 유지하되 명시적으로 reset

Offline generation path는 항상 scheduler `SequenceId`를 갖고 있지 않다. 이 경로는 model-owned fallback slot을 사용한다. Fallback은 fresh multi-token prefill이 들어올 때 reset되고, one-token decode step에서는 유지된다. 따라서 기존 single-stream API를 보존하면서 새 prompt가 이전 hidden state를 물려받는 문제를 막는다.

### 3.3 Missing scheduler state는 fail closed

Server scheduler는 sequence state를 allocate한 뒤 prefill/decode 전에 `prepare_sequence_state()`를 호출한다. `forward_with_sequence_id(Some(id), ...)` 호출 시 prepared slot이 없다면 lifecycle corruption이다. PR #1513은 scheduler path에서 새 `with_existing_sequence_state()` helper를 사용해 legacy single-stream fallback slot으로 조용히 떨어지지 않게 했다.

### 3.4 EOS는 checkpoint metadata에서 읽기

RWKV7, RecurrentGemma, KimiLinear는 hardcoded EOS id를 반환하고 있었다. 이제 model은 resolved EOS id를 저장한다. Direct config construction은 `eos_token_id`를 사용할 수 있고, 일반 directory load와 special weight load는 sidecar metadata가 있으면 `read_eos_token_ids()` 값으로 덮어쓴다.

## 4. Correctness review

- 영향 family의 `forward()`는 더 이상 fresh recurrent cache를 만들지 않는다.
- `make_caches()`는 fallback single-stream model-owned state만 reset하고 trait compatibility용 placeholder KV cache를 반환한다.
- `prepare_sequence_state()`는 fresh per-sequence cache vector를 넣는다.
- `forward_with_sequence_id(Some(id), ...)`는 prepared vector를 요구하고 호출 사이에 state를 갱신한다.
- `release_sequence_state_by_id()`는 per-sequence vector를 제거하므로 cancellation/completion 후 model-owned state가 남지 않는다.
- Layer/cache `zip()` 전에 cache cardinality를 assert해 malformed state가 layer 일부를 조용히 skip하지 못하게 했다.
- Scheduler inspection에서 allocation이 여전히 `prepare_sequence_state()`를 호출하고 completion/cancellation cleanup이 `release_sequence_state_by_id()`를 호출하는 것을 확인했다.

## 5. Security review

새 외부 process 실행, file deletion, credential handling, SQL, network request construction, web rendering 경로는 추가하지 않았다. 새 sidecar reader는 선택된 model directory 내부의 JSON file만 읽는다. EOS metadata가 없거나 malformed JSON이면 panic하지 않고 empty list를 반환하거나 다음 metadata source로 fallback한다.

명시적인 missing-state failure는 의도된 동작이다. Scheduler sequence 사이에서 fallback recurrent state를 공유하면 request-local generation history가 사용자 간에 섞일 수 있다. Corrupted hidden state로 generation을 계속하는 것보다 fail closed가 안전하다.

## 6. Performance review

이번 변경은 RWKV7, RecurrentGemma, KimiLinear에서 decode step마다 recurrent cache를 반복 allocation하던 문제를 제거한다. Persistent model-owned state는 token마다 clone을 추가하지 않는다. Sequence map에서 vector 하나를 빼서 mutate한 뒤 다시 넣는다. EOS sidecar check는 load/config resolution 시점에만 실행되며 decode hot path에는 없다.

이 PR은 세 family의 batching을 enable하지 않는다. `supports_batching() == false`를 유지하고 `SequenceStateLayout::model_owned(...)`를 광고하므로 scheduler는 계속 single-sequence model-owned execution을 사용해야 한다.

## 7. Validation record

| Check | 결과 | Notes |
|-------|------|-------|
| `cargo fmt --all --check` | Pass | 구현 후 formatting check. |
| `cargo test --lib model_owned_sequence_state -- --nocapture` | Pass | 13 passed, 7309 filtered. 새 generic/family state-continuity filter 포함. |
| `cargo test --lib read_eos_token_ids -- --nocapture` | Pass | 6 passed, 7316 filtered. generation/tokenizer/config EOS precedence와 tokenizer `eos_token` resolution 검증. |
| `cargo test --lib rwkv7_ -- --nocapture` | Pass | 4 passed, 7318 filtered. RWKV7 cache snapshot, EOS parsing, continuity, missing-state rejection 검증. |
| `cargo test --lib recurrent_gemma_ -- --nocapture` | Pass | 4 passed, 7318 filtered. RecurrentGemma EOS parsing, continuity, missing-state rejection, 기존 windowed attention classification 검증. |
| `cargo test --lib kimi_linear_ -- --nocapture` | Pass | 7 passed, 7315 filtered. 기존 KimiLinear guard와 새 EOS parsing, continuity, missing-state rejection 검증. |
| `cargo test --lib parse_eos_token_ids -- --nocapture` | Pass | 3 passed, 7319 filtered. Scalar/array/invalid EOS field parsing 검증. |
| `git diff --check` | Pass | Whitespace error 없음. |
| Static old-pattern grep | Pass | 세 target model file에 fresh-per-forward cache allocation과 hardcoded `[0]`, `[1]`, `[2]` EOS pattern이 남아 있지 않음. |

## 8. Validation limits

- 실제 RWKV7 checkpoint generation은 실행하지 않았다.
- 실제 RecurrentGemma checkpoint generation은 실행하지 않았다.
- 실제 KimiLinear checkpoint generation은 실행하지 않았다.
- Hardware-specific MLX generation qualification은 실행하지 않았다.
- Unit constraint에 따라 broad workspace test, broad `cargo test --lib`, broad workspace clippy, serial all-tests, release build는 실행하지 않았다.

## 9. 후속 작업

- Pinned RWKV7 checkpoint로 CLI greedy generation을 실행하고 prior context 변경에 따라 multi-token output이 달라지는지 확인한다.
- Pinned RecurrentGemma checkpoint로 CLI/server single-stream generation과 cancellation cleanup을 확인한다.
- Pinned KimiLinear checkpoint로 CLI/server single-stream generation과 두 개 concurrent queued request의 state isolation을 확인한다.
- 이 family들이 나중에 batched decode에 opt-in한다면 `supports_batching()` 변경 전에 명시적인 batched routing test를 추가한다.

## Appendix

- Issue: #1220
- PR: #1513
- Branch: `fix/issue-1220-recurrent-state-lifecycle`
