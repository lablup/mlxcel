# 기술 보고서: PR #1482 - Locally typical sampling (typical_p)

**작성일**: 2026-08-28

**상태**: 완료

**언어**: Rust

**위험도**: Medium

## 요약

#1375 row-filter 훅 위에 locally typical sampling(`typical_p`, 기본값 `1.0` 비활성)을 추가했다. surprisal이 행 엔트로피에 가장 가까운 토큰부터 typicality 순서로 `typical_p` 확률 질량이 쌓일 때까지 남긴다. 앞의 두 필터와 달리 이 이슈가 소유한 llama-compat 매니페스트 엔트리 2건(`--typical`, `field:typical_p`)을 `supported`로 플립했고, 그 때문에 두 서버 바이너리의 서버 전역 플래그와 네이티브 `/completion` 필드가 같은 PR에 들어왔다.

## 1. 문제 정의

mlxcel에는 typical 샘플러가 없었고, `--typical` / `field:typical_p` 엔트리는 이슈 #1377 소유다. 에픽의 종료 규칙상 이슈가 닫히기 전에 소유 엔트리 전부가 `deferred`를 벗어나야 하고, llama-compat 게이트는 supported 옵션 주장을 두 서버 바이너리에, supported 필드 주장을 `NativeCompletionRequest`에 묶는다. 따라서 엔진 의미론, 두 플래그 표면, 네이티브 필드, 매니페스트 플립이 한 변경에 함께 들어가야 했다.

## 2. 기술적 선택과 그 이유

### 2.1 체인 위치: 멱등 top-k 프리마스크

리뷰가 핵심 미묘함을 잡았다. b10621의 체인은 `top_k -> typ_p`이고 typical 샘플러는 생존 후보를 다시 softmax하므로 엔트로피가 재정규화된 top-k 분포에서 나온다. mlxcel 훅은 C++ 체인의 top_k보다 앞서 실행되어 처음에는 전체 어휘로 엔트로피를 계산했고, 서버 기본 top_k가 40이라 사실상 모든 활성 요청이 관찰 가능하게 갈라졌다. 수정은 `1 < top_k < vocab`일 때 typical 필터 앞에 top-k 마스크를 먼저 적용한다. top-k 마스킹은 멱등이라 C++ 체인의 top_k가 같은 집합을 다시 고르며 어떤 단계도 순서가 어긋나지 않는다. 두 순서가 증명 가능하게 갈라지는 행(전체 어휘 기준은 토큰 1만, top-k 재정규화 기준은 토큰 0만 남는다)으로 회귀 테스트를 추가했다.

### 2.2 표면별 값 도메인 분리

OpenAI 형태 엔드포인트는 `(0.0, 1.0]`을 검증하고 벗어나면 400을 반환한다. 네이티브 `/completion` 필드는 b10621의 무제한 스키마를 따라, 도메인 밖 값이 오면 명시적 비활성 `1.0`으로 해석하되 서버 기본값은 여전히 덮어쓴다. 서버 전역 `--typical` 플래그는 b10621처럼 받아들이되 시작 시 도메인 밖 값을 경고와 함께 비활성으로 접고, `/props`가 해석된 값을 보고해 운영자가 읽어볼 수 있다. CLI generate/chat 플래그는 파싱 시점에 거부한다.

### 2.3 유효값 정규화와 serde 기본값

`FusedSampleParams::from_config`와 speculative 윈도우의 `sampling_config_eq`는 `effective_typical_p()`(greedy와 도메인 밖 값은 `1.0`으로 접힘)를 비교하므로, 필연적으로 동일한 행들이 fused batch, pipelined lookahead, 공유 speculative 윈도우에 남는다. 분산 와이어 구조체는 serde 기본값 함수로 `1.0`을 쓴다. f32의 0 기본값이면 구버전 피어의 프레임이 항상 켜진 잘못된 컷오프가 되기 때문이다.

## 3. 변경 요약

| 항목 | 값 |
|-----|---|
| 변경 파일 | 35개 (리뷰 후속 +8) |
| 라인 | 두 커밋 합산 +884 / -33 |
| 테스트 추가 | 25개 |

검증: 40케이스 f64 호스트 레퍼런스 테스트를 포함해 코어 sampling 테스트 91개 green, compat 통합 테스트가 새로 빌드한 두 바이너리에서 플립된 `--typical` 주장을 검증, 매니페스트 체커 green(#1377 소유 deferred 엔트리 0건), Qwen3-4B-4bit 실체크포인트 검증(`typical_p 0.5`에서 `top_k 40` 유무와 무관하게 유창, greedy 대조군 토큰 동일).

## 4. 후속 조치

- #1373(`p_less`)이 `apply_row_filters`의 남은 슬롯을 채운다.
- #1436이 나머지 b10621 샘플링 의미론을 마무리한다. `typical_p_filter`(마스킹)와 `top_n_sigma_filter`(유지)의 `+inf` 처리 차이는 필터 문서에 기록해 두었다.
