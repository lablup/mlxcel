# 기술 보고서: PR #1507 - 모델 capability 기반 boundary snapshot gate

**작성일**: 2026-08-30
**상태**: 완료
**언어**: Rust
**위험도**: Medium

## 요약

PR #1507은 worker thread가 로드한 모델의 snapshot-reuse capability를 HTTP request preparation까지 전달하고, dense-KV 모델에서는 history-boundary render를 건너뛰게 한다. Snapshot-capable 모델은 기존 boundary path를 유지하며, worker가 `loaded=true`를 publish하기 전 startup 구간은 의도적으로 fail-open 처리해 첫 eligible request가 boundary snapshot을 잃지 않게 한다.

## 1. 문제 정의

Issue #1153은 prompt-cache chat path의 낭비를 지적했다. Text-only chat request는 dense-KV 모델에서도 conversation을 두 번 render하고 두 번째 tokenization을 queue했지만, scheduler는 이런 모델에서 model-owned recurrent snapshot을 사용할 수 없다.

기존 end-to-end opt-out은 `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`뿐이었다. 이 스위치는 알고 있는 operator에게는 유효하지만, 이미 loaded model trait이 가진 capability 답을 server가 자동으로 쓰지는 못했다.

이 path를 gate하지 않으면 긴 dense-KV conversation마다 추가 Minijinja render, `String` clone, tokenizer encode 비용을 지불하지만 cache 이득은 없다.

## 2. 기술적 선택과 그 이유

### 2.1 Provider를 통한 trait result publish

`ModelProvider`는 기존 `loaded` flag 옆에 `snapshot_reuse_capable`을 가진다. Batch/legacy worker는 model construction 직후, `loaded=true` 전에 `model.supports_snapshot_reuse()`를 저장한다. XLA worker는 model-owned snapshot reuse path를 쓰지 않으므로 `false`를 저장한다.

이 방식은 중복 static family table을 만들지 않는다. Scheduler가 나중에 참조하는 동일 trait method를 따르므로, future model addition은 model implementation만 갱신하면 된다.

### 2.2 Scheduler adoption이 아니라 request preparation에서 gate

`prepare_chat_request_with_cache`를 호출하는 모든 HTTP path는 별도의 `snapshot_reuse_capable` boolean을 넘긴다. History render는 prompt cache가 enabled이고, 모델이 snapshot을 reuse할 수 있고, request가 text-only이며, manual kill switch가 꺼져 있을 때만 실행된다.

Disaggregated router는 계속 text-only front이며 `false`를 넘기므로 single-node snapshot boundary work를 지불하지 않는다.

### 2.3 Readiness 전 fail-open

`AppState::should_render_history_boundary_snapshot()`은 worker가 provider를 loaded로 표시하기 전까지 true를 반환한다. Readiness 이후에는 worker가 publish한 trait result를 따른다. 이 선택은 issue #1153의 ordering 요구사항을 보존한다. Startup 중 admit된 request가 나중에 snapshot-capable로 판명되는 모델의 첫 boundary snapshot을 건너뛰지 않는다.

## 3. 변경 요약

| 카테고리 | 변경 수 | 주요 내용 |
|---|---:|---|
| Capability plumbing | 1 | `ModelProvider`, worker constructor, test fixture에 `snapshot_reuse_capable` 추가. |
| Request preparation | 1 | `prepare_chat_request_with_cache`와 모든 caller에 capability gate 전달. |
| Tests | 3 | History-render attempt instrumentation, dense-KV no-render coverage, snapshot-capable prefix coverage 유지. |
| Documentation comments | 2 | `PreparedChatRequest::history_prompt`와 state accessor comment에 새 gate와 startup ordering 설명 추가. |

## 4. 검증

- `cargo test --lib history_render_is`: 5 passed.
- `cargo test --lib prompt_cache_on_`: 4 passed.
- `cargo test --lib single_stream`: 5 passed.

이 Linux worktree에서는 실제 llama-3.2-1b-instruct 및 qwen3.5-0.8b-4bit benchmark parity를 실행하지 않았다. 이 PR은 targeted unit test로 code path를 검증하며, checkpoint counter parity는 적절한 hardware/model 환경에서 따로 검증해야 한다.

## 5. 리뷰 메모

- **Correctness**: Capability write는 release ordering으로 `loaded=true` 전에 발생하고, route read는 provider accessor를 통해 이뤄진다. 따라서 readiness ordering을 유지하면서 readiness 이후 false negative capability read를 피한다.
- **Security**: 새 request data 노출은 없다. 추가 flag는 process-local model capability이며 prompt나 user metadata가 아니다.
- **Performance**: Dense-KV text-only request는 추가 history render와 그 뒤 history-tokenization path를 건너뛴다. Snapshot-capable 모델은 기존 동작을 유지한다.
- **Compatibility**: Public API 변경은 없다. Route behavior는 소비할 수 없는 internal optimization artifact를 만들지 않는 방향으로만 바뀐다.

## 6. 후속 조치

- 적절한 machine에서 issue #1153의 real checkpoint matrix를 실행한다: llama dense-KV no-render counter, qwen3.5 snapshot hit parity, 문서화된 history-boundary benchmark number.
- 향후 snapshot-capable model 추가도 HTTP-side family table이 아니라 `LanguageModel::supports_snapshot_reuse()`에 연결한다.
