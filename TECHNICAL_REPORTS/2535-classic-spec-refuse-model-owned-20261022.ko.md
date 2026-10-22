# PR #2535: classic speculative walk에서 model-owned 패밀리 거부

**작성일**: 2026-10-22
**상태**: 완료
**위험도**: Medium
**언어**: Rust

---

## 요약

deprecated 상태인 classic `--draft-model` speculative walk는 거부된 draft 블록을 두 모델의 외부 KV 캐시를 trim하여 버린다. 시퀀스 상태가 다른 곳에 있는 패밀리(Qwen3-Next의 recurrent GDN 레이어, model-owned 캐시)에서는 이 trim이 아무것도 되돌리지 못하므로 이후 라운드가 오래된 위치를 기준으로 진행되고, `can_trim_prompt_cache` assert도 placeholder 캐시에서 자명하게 통과한다. 이 PR은 이슈 #2495의 수정안 1을 택한다: `SpeculativeGenerator::generate`가 이런 모델을 상태를 조용히 망가뜨리는 대신 시작 시점에 명시적 에러로 거부한다. 사용자에게 보이는 동작 변경: 공유 규칙이 padded prefill을 사용하지 않는 패밀리(Gemma 4, Llama 4)도 거부하므로, 이 모델들은 classic 경로에서 이전에는 실행되었으나 이제 에러가 발생한다. 실제 model-owned 체크포인트를 CLI로 실행해 검증하지는 않았다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #2495는 PR #2493(Qwen3-Next multi-token cache-forward 수정) 리뷰에서 발견되었다. 거부 경로는 generator의 외부 KV 캐시만 trim하는데, model-owned 패밀리는 이를 무시한다. 이 경로는 deprecated(#2176)이며 저장소 CLI 흐름에서는 프로세스당 한 번만 생성하지만, 라이브러리 사용자가 model-owned 모델에서 `SpeculativeGenerator`를 여러 프롬프트에 재사용하면 문제가 발생할 수 있다.

### 1.2 기존 문제점

- 첫 거부 라운드 이후 내부 GDN/KV 상태에 거부된 토큰이 남고, 이후 라운드가 오래된 위치를 기준으로 진행된다.
- `can_trim_prompt_cache` assert가 placeholder 외부 슬라이스에서 통과하므로 fail-fast가 없었다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| model-owned 패밀리에서 상태가 조용히 손상 | Medium (라이브러리 호출자에게 잘못된 speculative 출력) | 해당 모델에서 라이브러리 재사용 시 |
| Gemma 4 / Llama 4가 classic 경로에서 거부됨 | Medium (이전에 실행되던 구성이 에러로 변경) | 해당 패밀리에 `--draft-model` 사용 시 결정론적 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

새로운 입력 표면과 unsafe 코드는 없다. 거부는 prefill이나 forward 이전에 일어난다.

### 2.2 성능 관점

`generate` 시작 시 모델당 규칙 평가 1회와 CLI의 이른 target 검사 1회뿐이다. hot path 변경은 없다.

### 2.3 호환성 및 의존성

- **Breaking Changes**: `generate`가 `Result<_, SpeculativeGenerateError>`를 반환하므로 라이브러리 호출자는 이를 처리해야 한다(해당 타입은 이미 deprecated, #2176). CLI 동작 변경은 padded prefill을 사용하지 않는 패밀리(규칙의 doc comment 기준 Gemma 4, Llama 4)가 classic 경로에서 거부된다는 점이다.
- **새 의존성**: 없음 (새 에러 타입의 `thiserror` derive만 사용).

### 2.4 코드 품질

게이트는 `Engine::verify_rounds_unsupported`가 이미 적용하는 `prompt_lookup_unsupported_reason` 규칙을 재사용하며 두 번째 정의를 추가하지 않는다. 기존 `can_trim_prompt_cache` 및 빈 프롬프트 assert는 변경되지 않았다.

---

## 3. 기술적 결정

### 3.1 되감기 대신 거부

수정안 2(모델 인지형 `trim_state(None, n)`)는 Qwen3-Next가 `trim_state`를 override해야 하며, GDN 상태는 trim으로 정확히 되감을 수 없다. 이 PR은 거부를 택한다.

### 3.2 게이트 이전 reset

`generate`는 먼저 `reset()`을 호출한 뒤 target과 draft를 검사한다. reset은 generator 자신의 캐시와 카운터만 건드리므로, 재사용된 generator가 거부된 호출 뒤에 이전 성공 호출의 토큰을 유지하지 않는다.

### 3.3 CLI 에러 문구

CLI는 지원되지 않는 target을 draft 체크포인트 로드 이전에 거부하고, `generate`의 거부(draft 포함)도 같은 문구로 변환한다. 메시지는 `--draft-model` 없이 일반 생성을 안내하며, 같은 모델을 거부하는 `--prompt-lookup`이나 engine drafter는 권하지 않는다.

---

## 4. 구현 상세

- `src/lib/mlxcel-core/src/speculative/mod.rs`: 새 `SpeculativeGenerateError`(`TargetUnsupported`, `DraftUnsupported`, 사유 문자열 포함), `generate`의 게이트, 반환 타입과 조기 반환을 `Ok(...)`로 변경, 테스트용 mock `StateKind`(Dense, Recurrent, ModelOwnedLayout, NoPaddedPrefill)와 forward 호출 카운터.
- `src/lib/mlxcel-core/src/speculative/distribution_tests.rs`: 테스트가 `Ok`를 `expect`하도록 변경.
- `src/commands/generate.rs`: draft 로드 이전의 target 검사, 공유 `classic_speculative_refusal`, `--draft-kind` 에러 문구에 classic 경로가 해당 target을 거부한다는 설명 추가.
- `src/lib.rs`: `SpeculativeGenerator` 옆에 `SpeculativeGenerateError` 재export.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| `cargo test --profile test-fast -p mlxcel-core --lib speculative::` | 178 passed |
| `cargo test --profile test-fast -p mlxcel --bin mlxcel commands::generate` | 101 passed, 1 ignored |
| Revert 검사 (두 게이트 호출을 `None`으로 교체) | `speculative::tests`: 15 passed, 3 failed (새 테스트 3개), 이후 복원 |
| clippy `-D warnings` (mlxcel-core lib+tests, mlxcel lib+bin+tests) | clean |
| `cargo fmt --all --check` | clean |

새 테스트: `generate_refuses_a_model_owned_target_before_any_forward`와 `generate_refuses_a_model_owned_draft_before_any_forward`는 규칙의 세 가지 사유(배치 디코드 없음, 배치 가능 모델의 model-owned 레이아웃, padded prefill 없음)를 순회하며 에러가 거부된 모델을 규칙의 사유와 함께 명시하는지, 어느 mock의 `forward`도 실행되지 않았는지 확인한다. `refused_call_clears_tokens_from_a_previous_success`는 dense 호출 뒤 거부 호출을 한 generator에서 실행한다.

**미검증**: 실제 Qwen3-Next(또는 다른 model-owned 체크포인트)를 `mlxcel generate --draft-model`로 실행하지 않았다. 거부는 `SpeculativeGenerator::generate` 진입점에서 mock 모델로만 검증했고, CLI의 이른 target 검사는 컴파일과 코드 읽기로만 확인했다. `LoadedModel` 래퍼는 `supports_batching`, `sequence_state_layout`, `supports_padded_prefill`을 모델에 위임한다(읽기만 했고 실행하지 않음). revert 검사에서 각 거부 루프는 처음 실패하는 상태 종류에서 멈추므로 해당 케이스만 실패가 관찰되었다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 4 |
| 라인 | +324 / -103 |
| 추가 테스트 | 신규 3개 및 distribution 테스트 갱신 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|---------|
| `8f585e46` | fix | fix(speculative): refuse model-owned families in the classic walk |
| `751adfd1` | fix | fix(speculative): refuse unsupported targets before loading the draft |

---

## 7. 후속 조치

### 향후 개선

- 실제 model-owned 체크포인트(Qwen3-Next)와 Gemma 4 또는 Llama 4 쌍을 `mlxcel generate --draft-model`로 실행해 CLI 에러 문구를 end to end로 확인.
- classic 경로는 v0.8.0에서 제거 예정(#2176)이며, 이 거부는 그때까지의 임시 조치다.

### 모니터링 필요

- Gemma 4 또는 Llama 4 사용자가 `--draft-model`에서 새 거부에 부딪혔다는 보고.
