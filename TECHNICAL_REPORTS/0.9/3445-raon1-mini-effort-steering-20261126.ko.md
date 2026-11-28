# PR #3445: Raon-1 mini reasoning-effort steering (이슈 #3435)

**작성일**: 2026-11-26

**상태**: PR 열림(#3435를 닫음). M3 Ultra에서 검증. 63개 파일, +2103 / -116.

**위험도**: 중간(`LanguageModel`과 스케줄러에 시퀀스별 채널 추가, 다른 모델 계열에는 no-op)

## 요약

`KRAFTON/Raon-1-mini` 정식 릴리스는 12, 14, 16, 18, 20번 레이어의 MoE 출력에 고정된 f32 벡터를 더하고, 그 크기를 요청의 reasoning effort에 따른 alpha로 정한다. 기본 effort(`high`, alpha 0.05)가 0이 아니므로, #3225 포트(alpha 0)는 `max`를 요청하지 않은 모든 요청에서 reference와 다른 logit을 냈다. 이 PR은 steering을 적용하고, 각 요청의 effort를 서버와 스케줄러를 거쳐 그 요청의 배치 행까지 전달하며, prompt cache 키에 effort를 넣는다.

Greedy 출력은 기본, `low`, `none` effort에서 Transformers reference와 일치한다. 871토큰 프롬프트의 teacher-forced trace에서 결정 위치 불일치는 기본 0 / 541, `low` 0 / 542이다.

## 1. 문제 정의

### 1.1 배경

정식 체크포인트와 프리뷰의 차이는 `moe.reasoning_effort_steering` 텐서 다섯 개와 `reasoning_effort_steering` config 블록(레벨별 `alpha`, `default: "high"`, 0부터 세는 `layers`)뿐이다. Reference는 이미 반올림된 MoE branch에 `alpha * c`를 f32로 더하고 mHC `mix` 전에 한 번 반올림하며, alpha가 0이면 더하기를 건너뛴다.

### 1.2 기존 문제점

- **모델 쪽 effort 채널이 없었다.** `chat_request.rs`의 `map_reasoning_control_kwargs`는 템플릿이 `reasoning_effort`를 언급할 때만 그 값을 넘겼다. Raon 템플릿은 언급하지 않아 레벨이 버려졌다.
- **effort는 모델 단위가 아니라 요청 단위다.** 벤더 MLX 런타임은 캐시별로, vLLM 포크는 요청별로 둔다. 그래서 batched decode에는 행별 alpha가 필요하다.
- **thinking과 steering은 별개의 제어다.** 템플릿은 `enable_thinking`만 읽고, steering은 프롬프트를 바꾸지 않는다. 모델 카드는 `reasoning_effort: "none"`과 `enable_thinking: false`가 같은 결과를 내야 한다고 명시한다.

## 2. 기술적 선택과 그 이유

### 2.1 행별 alpha를 `[B, 1, 1]` 배열로

| 선택지 | 장점 | 단점 |
|---|---|---|
| 모델 전역 effort setter | 변경이 작다 | effort가 다른 동시 요청에서 틀린다 |
| **시퀀스별 alpha를 `[B, 1, 1]` f32로 broadcast** | prefill, native batched decode, 행별 fallback이 한 코드 경로 | trait와 스케줄러를 관통하는 시퀀스별 채널이 필요 |

`Raon1MiniLM`은 `EffortState`(`raon1_mini_effort.rs`)를 가진다. 모델 기본값과 `SequenceId`별 맵이다. steering 블록이 없거나 모든 alpha가 0이면 `steer_rows`는 `None`을 돌려주므로, alpha 0 경로는 MLX 배열을 만들지 않고 host sync도 추가하지 않는다. steered 배치 안의 alpha 0 행은 비트 단위로 그대로다. bf16에서 f32로, 다시 bf16으로 가는 왕복은 손실이 없고 0을 더해도 값이 바뀌지 않기 때문이다.

### 2.2 Raon 전용 hook이 아닌 범용 trait 채널

`LanguageModel::set_sequence_reasoning_effort`와 `sequence_reasoning_effort`의 기본 구현은 `Ok(())` / `None`이다. 스케줄러는 admission 시점(prefix adoption 뒤, prefill 전)에 이를 호출하므로 잘못된 레벨은 그 요청만 실패시킨다. preemption과 queued adoption에서는 레벨을 새 시퀀스 id로 옮긴다. 시퀀스 id를 prepare하거나 release하면 레벨이 지워지므로, 재사용된 id가 다른 요청의 effort를 물려받지 않는다.

### 2.3 렌더, steering, 캐시 키에 하나의 resolver

`resolve_model_reasoning_effort`(`src/server/reasoning_steering.rs`)는 모델 `config.json`에 steering 블록이 있을 때만 동작한다. 순서는 프롬프트 렌더의 thinking 토글을 따르므로 프롬프트와 alpha가 어긋날 수 없다.

1. 요청이 명시한 레벨(`chat_template_kwargs.reasoning_effort`, 최상위 `reasoning_effort`, Responses `reasoning.effort`).
2. 요청 자체의 thinking-off(`enable_thinking: false`, `reasoning: false`)는 `none`.
3. 서버 기본 레벨(`--reasoning-effort`, `--chat-template-kwargs`).
4. 서버 기본값의 thinking-off는 `none`.
5. config 기본값(`high`).

이 순서는 리뷰 지적 두 건으로 정해졌다. 첫 버전은 명시적 kwargs와 서버 기본값만 읽어서, 서버 기본값이 thinking-off이고 요청이 `reasoning: true`를 보내면 thinking은 켜진 채 렌더되는데 steering은 `none`이 됐다(`ddf59049b`에서 수정). 또 서버 기본 레벨이 요청 자체의 `enable_thinking: false`를 이겼다(`1ebc2ce34`에서 수정). portable `none`은 `<think></think>`도 렌더한다. `chat_template_kwargs` 안의 `none`은 모델 카드대로 alpha만 바꾼다.

결정된 레벨은 prompt cache 템플릿 시그니처에 접힌다. 레벨이 없으면 시그니처가 바뀌지 않아 다른 계열의 캐시 버킷은 그대로다.

## 3. 구현 상세

- `src/models/reasoning_effort_steering.rs`: 블록을 파싱하고 검증한다. `none`은 config의 `none` alpha가 0보다 크면 그 값, 아니면 `NO_THINKING_ALPHA = 0.1`이다. `default: "none"`, 음수이거나 f32로 넘치는 alpha(`0d425b857`: `1e39`는 f64에는 들어가지만 forward에서 `inf`가 됐다), 범위 밖·중복·MoE가 아닌 레이어는 로드 시 이름이 붙은 오류로 실패한다. 벡터가 없거나 `[hidden_size]`가 아니어도 이름이 붙은 로드 오류다.
- `raon1_mini*.rs`: 나열된 레이어의 MLP 위치에서 prefill(`forward_rows_for_sequence`), native batched decode, 행별 fallback 모두에 steering을 적용한다.
- CLI: `mlxcel generate --reasoning-effort`는 모델 기본값을 정하고, 30 GB 가중치를 올리기 전에 `config.json`으로 검증한다. `examples/logit_trace`는 `MLXCEL_TRACE_REASONING_EFFORT`를 읽고, `scripts/raon1_mini_oracle_trace.py`는 `--effort`를 받는다.
- 문서: `docs/supported-models.md`, `docs/responses-api.md`, `CHANGELOG.md`, `mlxcel arch` 표시 이름.

## 4. 검증

Reference: 체크포인트의 `modeling_raon1_mini.py`(Transformers, bf16, CPU). 프롬프트: 22토큰, 871토큰, thinking off.

| 확인 | 결과 |
|---|---|
| `--reasoning-effort none`, "17 + 25" | `42` 후 EOS |
| Greedy 64, 짧은 프롬프트, 기본 / low / none | reference와 동일 |
| Greedy 64, 긴 프롬프트, 기본과 low | 텍스트 동일, low는 max와 다름 |
| Teacher-forced, 긴 프롬프트, 기본 / low | 결정 위치 불일치 0 / 541, 0 / 542 |
| Thinking on, 기본, 짧은 프롬프트 | 생성 4번째 토큰에서 갈라짐. reference 텍스트 기준 결정 위치 불일치 0 / 30, flip gap 0.75. 둘 다 42 |
| `max` 대 변경 전 바이너리 | 텍스트 동일, 비트 일치는 tiny 모델로 증명 |
| 서버, low와 max 동시 | 각각 단독 실행 출력과 같음 |
| 서버, `minimal` | 허용 집합을 나열한 400 `invalid_request_error` |

단위 테스트는 alpha 결정, tiny 모델 steering(건너뛰면 실패, `max`는 unsteered와 같음), 행별 batched decode 대 단독, 시퀀스 lifecycle, 스케줄러 admission과 preemption 이월, 모든 서버 채널과 우선순위, 캐시 키를 다룬다. 로컬 `make verify` 전체 게이트를 통과했다.

## 5. 후속 조치

- 서버는 Raon에서 동일한 반복 요청에도 prompt cache prefix를 재사용하지 않는다. 이 PR 전후 모두 그렇다. Raon에 모델 소유 snapshot/restore가 없는 것이 원인으로 보인다. 그래서 캐시 키의 effort 분리는 store 테스트로만 검증됐다.
- Queued adoption은 Raon 시퀀스를 고르지 않는다(paged block 없음). 이 계열에서 그 경로의 effort 이월은 테스트되지 않았다.
- 분리형 prefill handoff와 pipeline/XLA generate 경로는 effort를 무시한다. 현재 Raon은 이 경로를 타지 않는다.
- Thinking on은 기본 effort에서만 reference와 비교했다. CUDA와 ROCm은 실행하지 않았다.
