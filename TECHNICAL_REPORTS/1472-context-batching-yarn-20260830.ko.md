# 기술 보고서: PR #1503 - b10621 컨텍스트 유지 정책, YaRN 오버라이드, 배칭 분류

**날짜**: 2026-08-30

**상태**: 완료

**언어**: Rust

**위험도**: 높음

## 요약

#1450이 `compat/llama-server/b10621/runtime-and-context.json`에 남긴 13개 `deferred` 엔트리를 종결한다(#1472). 핵심 구현은 두 가지다. 공유 RoPE 경로에 YaRN 암을 추가해 `--rope-scaling yarn`과 다섯 개 `--yarn-*` 노브가 실제로 회전을 조정하게 했고, b10621의 컨텍스트 유지 계약(`--context-shift` 기본 비활성, `--keep`, 요청별 `n_keep` / `n_discard`)을 구현하면서 mlxcel의 무조건적 무음 KV 앞단 트리밍을 의도적으로 대체했다. `--parallel`은 업스트림 auto와 동일하게 4 슬롯으로 해석되는 `-1` 기본값을 갖는다. 9개 엔트리가 `supported`, 1개가 `aliased`, 2개가 `not_applicable`로 이동했고, `--parallel`은 남은 단일 divergence를 소유한 #1473(`--kv-unified`)으로 재지정되어 `deferred`로 남는다.

## 1. 문제 정의

PR #1464는 플래그 수용 표면만 구현하고, 그 뒤의 동작이 존재하지 않음을 엔트리별로 기록했다. YaRN 값은 시작 시 거부되었고, 컨텍스트 유지는 고정 4토큰 어텐션 싱크의 무음 트리밍이었으며, `--parallel -1`은 usize 파서가 거부했고, `--swa-full` / `--ubatch-size` / `--batch-size`에는 최종 분류가 없었다.

## 2. 기술적 결정

### 2.1 주파수 테이블 + 크기 스칼라로서의 YaRN

`RopeScalingKind::Yarn { freqs, mscale }`은 트리 내 deepseek_v2 / 업스트림 `YarnRoPE` 수식을 ggml의 외삽 혼합으로 일반화해 이식했다. 온도 보정은 Q와 K를 회전 전에 곱하며 1.0에서는 곱을 건너뛰어 비-YaRN 그래프가 바이트 동일함을 트레이스로 증명했다. `--yarn-attn-factor`가 외삽 혼합 0에서만 반영되는 b10621 자체의 해석 순서(`llama-context.cpp`)까지 의도적으로 미러링했다. 공유 경로에서 체크포인트가 선언한 `yarn` 블록은 이제 경고 후 무시 대신 선언된 회전으로 동작하며, 이는 b10621과 mlx-lm `initialize_rope` 양쪽과 일치한다.

### 2.2 KV 경계를 무음 트리밍에서 계약으로

b10621의 3부 계약을 그대로 구현했다. 어드미션은 슬롯당 경계를 넘는 프롬프트를 400 `exceed_context_size_error`로 거부하고, 시프트 비활성 상태의 디코드는 경계에서 `truncated: true` / `stop_type: "limit"`(신규 `StopKind::ContextExhausted`)으로 정지하며, `--context-shift`는 해석된 유지값(요청 `n_keep` 우선, `-1` = 전체 프롬프트, BOS 검출 시 +1, `bound - 4`로 클램프; discard는 `n_discard` 또는 비유지 윈도우의 절반, 프리필 청크 초과분만큼 상향)으로 트리밍한다. 산술은 순수 헬퍼로 분리해 단위 테스트로 고정했다. 기본 경로 변경은 마이그레이션 노트로 문서화했고, `--context-shift --keep 4`가 기존 롤링 윈도우를 재현한다.

### 2.3 나머지 엔트리의 정직한 최종 상태

`--parallel -1`은 업스트림 auto와 동일한 4 슬롯으로 해석되며, 업스트림 auto의 `kv_unified` 절반은 해당 엔트리의 유일한 divergence로 기록되어 #1473을 따른다. `--batch-size`는 `--prefill-chunk-size`로 `aliased`(기본값 차이 기록), `--ubatch-size`는 `not_applicable`(통합 메모리에 물리 마이크로배치 없음), `--swa-full`은 선언 후 시작 시 거부한다(링 캐시는 모델 소유이고, 이 플래그가 업스트림에서 구매하는 상태 연산은 링 크기가 아니라 스케줄러 소유 캐시에 게이트된다).

## 3. 변경 요약

| 항목 | 값 |
|------|------|
| 변경 파일 | 64 |
| 매니페스트 | supported: --context-shift, --keep, field:n_keep, field:n_discard, --yarn-* 5종; aliased: --batch-size; not_applicable: --ubatch-size, --swa-full; #1473로 deferred: --parallel |

검증: `origin/main` 대비 teacher-forced 로짓 트레이스(qwen3-0.6b-4bit 무플래그와 체크포인트 선언 YaRN인 deepseek-v2-lite-4bit 모두 바이트 동일; 센티널 암 바이트 동일; 강제 yarn은 top-1 19.2% 분기, `--yarn-beta-fast` 8 대 32는 두 yarn 암을 9.0% 분리), 그리고 `--ctx-size 512` 실서버 검증(과길이 프롬프트 400, 기본 경로 경계 정지 `truncated: true`, `--context-shift --keep 32`로 512 윈도우를 통과해 요청한 800토큰 전부 생성).

## 4. 후속 작업

- #1473이 `--kv-unified`와 `--parallel`의 재지정된 divergence를 소유한다.
- Turbo 양자화 KV 레이어에서 컨텍스트 시프트는 기록된 no-op(시작 시 경고)으로 남고, VLM 시퀀스는 면제로 남으며, 둘 다 매니페스트 엔트리에 문서화되어 있다.
