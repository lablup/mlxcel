# 기술 보고서: PR #2204 - 행 단위 샘플링 단계 하나와 token bias 조합 하나

**작성일**: 2026-10-07

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust

**위험도**: 중간. 이제 행 단위 샘플링을 하는 모든 지점이 타입 하나를 거친다. 실제 체크포인트 두 개에서 greedy, seeded, 페널티, DRY, 언어 bias 설정 모두 CLI 출력이 변경 전후 바이트 단위로 같았고, CLI와 서버 경로의 출력도 계속 같다.

## 요약

epic #2166의 Phase 2(#2169)다.
- **`RowSampler`**(`mlxcel_core::sampling_row_step`)가 이제 양쪽의 행 단위 샘플링 단계를 맡는다.
  - 샘플러 상태 수명
  - #2090에서 다룬 이력 순서 불변식
  - 구조화 출력 마스크
  - thinking budget 덮어쓰기
- **호출 지점**: `CxxGenerator` 루프 네 개와 서버의 세 지점(배치 행 단위, `decode_single_step`, prefill 첫 토큰)이다.
- **token bias**: `compose_token_bias`가 CLI와 서버에 같은 우선순위를 적용한다.
- **fused 경로**: fused 배치 경로의 사용 조건을 같은 단계 목록에서 끌어내므로, 샘플러 단계를 새로 추가할 때 한쪽만 빠뜨리는 일이 생기지 않는다.

## 1. 문제 정의

CLI와 서버는 샘플러 상태를 각자 다르게 다뤘다.
- CLI는 이력을 읽는 샘플러에만 `SamplerState`를 만들고 `accept_token`을 부르지 않았기 때문에, mirostat과 adaptive-p 상태를 이어갈 수 없었다. CLI에서는 이 샘플러들을 켤 수 없어서 드러나지 않았을 뿐이다.
- #2090이 CLI 루프 네 곳에서 고친 토큰 이력 순서는 서버에 따로 구현되어 있었다.
- token bias(요청 bias, 언어 bias, 출력 억제)를 합치는 우선순위가 양쪽에서 달랐다.
- fused 경로 조건 `config_supports_fused_batch_except_bias`는 단계를 추가할 때마다 손으로 넓혀야 했다(#1485).

## 2. 변경 요약

- **`RowSampler`**
  - `needs_token_history() || needs_sampler_feedback_state()`이면 상태를 한 번만 만든다.
  - `needs_host_token_before_sample()`은 이력을 읽는 샘플러에서만 참이다. 그래서 피드백만 쓰는 샘플러는 호스트 읽기를 강제하지 않고도 상태를 유지한다.
  - 마스크는 샘플러 체인보다 먼저 적용되고, `resolve`는 thinking 덮어쓰기를 위해 샘플한 토큰과 최종 토큰을 함께 돌려준다.
  - 받아들인 토큰마다 `accept_token`을 호출한다.
- **`SamplerStage`**: fused 사용 조건을 이 enum에 대한 빠짐없는 match에서 끌어내고, lang-bias 카운터 조건도 계속 확인한다(#2188).
- **`compose_token_bias`**(`sampling_token_bias.rs`): 요청 bias를 먼저 적용하고, 요청 map이 비어 있을 때만 언어 bias를 적용하며, 마지막으로 출력 억제를 `-inf`로 덮어쓴다. 요청 bias가 `+inf`나 NaN이어도 이 억제를 넘을 수 없다.
- **윈도우 페널티 집계**(리뷰에서 나옴): `SamplerState::for_config`는 `penalty_last_n < 0`일 때만 전체 이력 집계를 유지한다. 서버 기본값 64를 포함한 윈도우 설정은 이제 읽지도 않는 집계에 프롬프트 전체를 쌓지 않는다.

## 3. 기술적 선택과 그 이유

**CLI의 파이프라인은 유지한다.** 샘플링 단계가 루프에 샘플링 전에 호스트 토큰이 필요한지 알려 준다. 그래서 CLI는 이력을 읽지 않는 샘플러에서는 지연 평가와 파이프라인 decode를 그대로 쓰고, 이력이 필요한 샘플러에서만 토큰을 먼저 읽는다. #2117에서 정한 방식 그대로다.

**서버의 bias 순서를 채택했다.** 이슈는 "출력 억제가 항상 이긴다"는 것 외에는 병합 순서를 열어 두었다. 요청들이 이미 서버 순서에 기대고 있으므로, CLI도 자기가 가진 bias 원천에 대해 그 순서를 따른다. 그 원천들에 대한 CLI 출력은 바뀌지 않았다.

## 4. 검증

- **단위 테스트**
  - mlxcel-core: `sampling` 200, `generate` 61, `decode_finish` 9, `sampler_state` 6, `sampling_token_bias` 5.
  - 루트 크레이트: `server::batch` 508(7개 무시), `sampling` 54.
  - 새 테스트: 윈도우 집계 경우, 그리고 출력 억제가 `+inf`, NaN, 누적 NaN bias보다 우선하는 경우.
- **GB10 실제 체크포인트**
  - qwen3-1.7b-4bit와 llama-3.2-1b-instruct-4bit에서 CLI 출력이 변경 전 커밋과 바이트 단위로 같았다. greedy, seeded(temperature 0.8, repetition penalty 1.15, DRY 0.8), 언어 bias가 6/6이고, 윈도우 페널티가 4/4다.
  - `mlxcel-engine-parity --no-prompt-cache-case --expect-identical`은 두 모델 모두 6쌍이 같았고, 변경 전 기준선과도 일치했다.
- **병합**: #2201을 브랜치에 병합했다. 충돌은 import에서만 났고, 테스트 fixture의 필드 이름 하나를 바꿨다.

## 5. 남은 위험

- CLI와 서버 출력 비교 단위 테스트는 서버 단계를 손으로 옮겨 쓴 복사본과 비교한다. 실제 스케줄러와의 비교는 engine-parity 하네스가 맡는다.
- 별도 이슈로 다룰 이전부터 있던 LOW 문제
  - `native_completion.rs`가 bias를 `n as f32`로 읽어서 `1e39`가 `+inf`가 된다.
  - handoff와 CLI 파이프라인 경로는 출력 억제를 건너뛴다. 실제로는 텍스트 전용이라 영향이 없다.

## 6. 학습 포인트

- **상태를 미리 만들면 비용이 따른다.** 상태 생성을 생성 시점으로 옮기자, 기본 윈도우 페널티는 읽지도 않는 전체 이력 집계가 드러났다. 리뷰가 그 토큰마다의 추가 작업을 잡아냈다.
- **사용 조건은 단계 목록에서 끌어낸다.** 단계 enum에 대한 빠짐없는 match를 쓰면, fused 경로 조건을 빠뜨렸을 때 컴파일 오류가 난다.

## 7. 관련 항목

- epic #2166, 이슈 #2169.
- ADR 0007.
- #2090 / #2117(이력 순서).
- #2188(lang-bias 카운터).
- #2201(종료 처리).
