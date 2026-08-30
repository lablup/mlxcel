# 기술 보고서: PR #1509 - 모델 인식 snapshot cache capacity

**작성일**: 2026-08-30
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Medium

## 요약

PR #1509는 issue #1167을 해결한다. 기존 implicit snapshot-cache capacity는 512 MiB 고정값이었지만, 이제 snapshot-capable 대형 hybrid model에 대해 startup에서 bounded model-aware default를 계산한다. Operator가 CLI/env로 명시한 capacity는 계속 우선한다. 또한 `/v1/cache/stats`에 per-entry snapshot byte와 same-session self-eviction counter를 추가하고, snapshot LRU가 방금 snapshot을 donate한 session을 evict할 때 session당 한 번 WARN을 남긴다.

## 1. 문제 정의

Qwen3.8-27B 4-bit는 agent-scale prompt에서 snapshot entry 하나가 기존 512 MiB default에 근접한다. 이 상태에서는 snapshot store가 새 entry를 accept한 직후 같은 session의 live entry를 LRU로 evict할 수 있고, 다음 turn은 재사용할 snapshot을 찾지 못해 multi-turn hit rate가 0%까지 떨어진다.

기존 증상은 진단하기 어려웠다. Counter에는 insert와 LRU eviction만 보이고, operator가 `snapshot_bytes / snapshot_entries`를 직접 계산해야 loaded model에서 fixed default가 사실상 capacity one이라는 점을 알 수 있었다. Issue는 model-owned snapshot family에서 `--kv-bits` / `--kv-cache-mode` 동작도 silent accept가 아니라 명시적으로 설명되길 요구했다.

## 2. 기술적 선택과 그 이유

### 2.1 Operator capacity 우선권 유지

`PromptCacheConfig`는 이제 `snapshot_capacity_bytes`가 CLI/env/builder에서 명시된 값인지 추적한다. Startup은 이 값이 false일 때만 model-aware default를 적용한다. 따라서 기존 deployment tuning을 보존하고, operator가 의도적으로 더 작거나 큰 cap을 설정한 경우 이를 덮어쓰지 않는다.

### 2.2 `config.json` 기반 implicit default 계산

새 snapshot sizing module은 model metadata를 읽고 기존 architecture-aware KV estimator를 재사용한다. Default는 `min(context_size, 8192)` token에서 representative snapshot 6개를 담을 수 있도록 계산한다. Snapshot serialization은 현재 non-FP16 sidecar를 저장하지 않으므로 FP16 기준을 사용한다. Qwen3-Next / Qwen3.5-family config에서는 linear-attention dimension에서 fixed gated-delta recurrent/conv state estimate도 더한다.

계산된 raise는 detected available memory의 1/4로 clamp된다. 이 방식은 large model의 multi-turn reuse 공간을 확보하면서도 metadata만으로 제한된 host memory 대부분을 소비하지 않게 한다.

### 2.3 정상 supersede와 capacity thrash 분리

`PromptCacheStats`와 `/v1/cache/stats`는 이제 `snapshot_self_evictions`를 노출한다. 이 counter는 snapshot insert를 admit하는 과정에서 LRU enforcement가 같은 session chain의 entry를 evict할 때 증가한다. Strict-prefix 기반의 정상 same-session replacement는 계속 `snapshot_supersedes`로 집계된다.

WARN duplicate suppression set은 distinct session key 4096개로 제한된다. 따라서 악의적이거나 우발적인 high-cardinality session identifier가 memory를 무한히 늘리지 못한다. 이 cap에 도달해도 public counter는 계속 정확하게 증가한다.

### 2.4 Snapshot byte sizing을 직접 노출

`/v1/cache/stats`는 이제 `snapshot_bytes_per_entry`를 포함한다. 값은 live snapshot이 없을 때 0이고, 그 외에는 `snapshot_bytes / snapshot_entries`이다. Operator sizing guide가 수동 나눗셈 없이 단일 field lookup으로 끝난다.

### 2.5 KV-mode contract 문서화

문서는 model-owned snapshot family가 non-FP16 attention KV sidecar를 조용히 snapshot entry에 serialize하지 않는다고 명시한다. 현재 해당 mode는 live attention layer에 적용되고 startup/log stats에 report되지만, sidecar snapshot support가 생기기 전까지 snapshot donation은 named warning과 함께 skip된다.

## 3. 변경 요약

| 카테고리 | 변경 수 | 주요 내용 |
|---|---:|---|
| Capacity sizing | 1 | Model-aware snapshot default 계산 및 startup wiring 추가. |
| Config authority | 1 | CLI/env/builder override에 대한 explicit-capacity tracking 추가. |
| Observability | 2 | Cache stats에 `snapshot_bytes_per_entry`, `snapshot_self_evictions` 추가. |
| Runtime warning | 1 | Same-session snapshot self-eviction에 bounded once-per-session WARN 추가. |
| Documentation | 3 | Env var 문서, Turbo KV 문서, issue #1167 validation record 업데이트. |
| Tests | 5 | Deterministic sizing, override, stats, self-eviction, supersede coverage 추가. |

## 4. 검증

- `cargo test --lib server::prompt_cache::snapshot_sizing::tests`: 3 passed.
- `cargo test --lib snapshot_capacity_self_eviction_is_counted_and_warned_once_per_session`: 1 passed.
- `cargo test --lib supersede`: 7 passed.
- `cargo test --lib cache_stats`: 3 passed.
- `cargo test --lib build_stats_response_reports_snapshot_bytes_per_entry`: 1 passed.
- `cargo test --lib prompt_cache_snapshot_limits`: 2 passed.
- `git diff --check`: passed.

이 Linux/aarch64 worktree에서는 issue #1167의 Apple Silicon / Metal Qwen3.8-27B benchmark를 재실행하지 않았다. PR은 해당 external evidence를 issue context로 보존하고, implementation behavior는 deterministic local test로 검증한다.

## 5. 리뷰 메모

- **Correctness**: Capacity가 explicit이면 model-aware sizing을 건너뛴다. Standard full-attention model은 snapshot-capacity raise를 받지 않는다. Qwen3-shaped test는 계산된 KV byte와 fixed-state byte를 고정한다.
- **Security**: WARN suppression map은 bounded이므로 high-cardinality session key로 인한 unbounded memory growth를 막는다. Prompt content나 token vector는 log에 남기지 않는다.
- **Performance**: Startup sizing은 `config.json`과 기존 memory estimator metadata만 읽는다. MLX model state나 tensor를 load하지 않는다. Hot insert overhead는 snapshot LRU enforcement 중 same-session key 비교로 제한된다.
- **Compatibility**: Stats response는 additive field만 추가한다. 기존 CLI/env knob 이름과 precedence는 유지된다.

## 6. 후속 조치

- 적절한 hardware에서 agent-scale Qwen3.8-27B Apple Silicon / Metal benchmark를 재실행하고 before/after hit-rate median을 기록한다.
- Model-owned non-FP16 snapshot sidecar serialization을 지원하게 되면 real checkpoint coverage를 추가한다.
