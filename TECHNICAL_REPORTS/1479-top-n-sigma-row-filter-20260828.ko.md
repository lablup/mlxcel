# 기술 보고서: PR #1479 - Top-n-sigma 로짓 필터와 row-filter 훅

**작성일**: 2026-08-28

**상태**: 완료

**언어**: Rust

**위험도**: Medium

## 요약

top-n-sigma 샘플러(`SamplingConfig::top_n_sigma`, 기본값 `0.0` 비활성)와, 이후 샘플러 단계(`typical_p` #1377, `p_less` #1373)가 꽂혀 들어갈 배치 안전 row-filter 훅 `apply_row_filters`를 추가했다. 이 필터는 원시 로짓이 행 최대값에서 `n` 표준편차 이내인 토큰만 남기므로 고온 샘플링을 안정시키면서 온도 값 자체에는 불변이다. 비활성 기본값은 비트 단위로 동일하다. 필터가 꺼져 있거나 greedy 설정이면 그래프 노드를 하나도 추가하지 않는다.

## 1. 문제 정의

mlxcel에는 top-n-sigma 샘플러가 없었고, RNG와 히스토리에 의존하지 않는 행 단위 로짓 필터를 fused batch 경로에서 이탈시키지 않고 실행할 공용 훅도 없었다. llama-server b10621은 요청 스키마에 `top_n_sigma`, CLI에 `--top-nsigma`를 노출하므로, 네이티브 철자를 매니페스트에서 주장(#1436)하기 전에 엔진에 의미론이 먼저 필요하다.

## 2. 기술적 선택과 그 이유

### 2.1 C++ 체인 단계가 아닌 Rust 프리필터

fused 샘플러 체인(temperature, top_k, top_p, min_p, XTC)은 `ffi::fused_sample` 뒤 C++에 있다. top-n-sigma는 그 단일 디스패치 앞에서 로짓에 적용되는 Rust 쪽 그래프 변환으로 구현했다. 난수를 쓰지 않고 스텝별 상태도 없으므로 MLX의 lazy 그래프가 같은 평가로 융합한다. 브리지 시그니처 변경도, MLX 핀 변경도 없고 `[B, V]` 배치 경로는 단일 디스패치를 유지한다.

### 2.2 유한 엔트리만으로 float32 통계 계산

15만 어휘에 대한 f16 합은 `inf`로 오버플로우해 필터를 조용히 무력화하므로, 로짓 dtype과 무관하게 모든 리덕션을 f32로 수행한다. 토큰 바이어스나 페널티로 이미 `-inf`가 된 엔트리는 평균, 표준편차, 최대값 계산에서 제외되고 출력에서도 마스크가 유지된다. 리뷰에서 두 가지 경계를 보강했다. 행 최대값을 유한 마스킹된 행 위에서 취해 NaN 하나가 행 전체를 `-inf`로 무너뜨리지 못하게 했고(MLX Max 리듀서는 NaN을 전파한다), 마스크 채움값이 원본 로짓 dtype을 유지하게 했다(MLX `where` 승격이 f16/bf16 입력을 f32로 바꾸고 있었다).

### 2.3 배치 게이트의 유효값 정규화

`FusedSampleParams::from_config`와 speculative 윈도우의 `sampling_config_eq`는 `SamplingConfig::effective_top_n_sigma()`를 비교한다. greedy, 0 이하, 비유한 값은 `0.0`으로 접힌다. 이 정규화가 없으면 inert한 `top_n_sigma`만 다른 두 greedy 요청이 바이트 동일 토큰을 내면서도 비트 비교에 실패해, 같은 배치 전체가 per-row 루프로 떨어지고 speculative 윈도우가 관찰 가능한 차이 없이 쪼개진다.

### 2.4 네이티브 `/completion`은 #1436으로 이관

`src/server/llama_compat_tests.rs`는 `native_request_field` 매니페스트 엔트리의 플립과 `NativeCompletionRequest` 필드 선언이 한 변경에서 함께 이루어질 것을 요구하고, `field:top_n_sigma` 엔트리는 에픽의 샤드 소유 규칙상 #1436 소유다. 따라서 필드는 매니페스트 플립과 함께 #1436에서 들어간다. `apply_row_filters`는 업스트림의 `-1.0` 비활성 센티널을 이미 inert하게 처리한다.

## 3. 변경 요약

| 항목 | 값 |
|-----|---|
| 변경 파일 | 27개 (리뷰 후속 +105줄 포함) |
| 라인 | 두 커밋 합산 +642 / -13 |
| 테스트 추가 | 17개 (코어 필터, 게이트, 요청 계층, 와이어 프로토콜, burst 윈도우) |

접점: `mlxcel-core`(`generate.rs`, `sampling.rs`), 스케줄러 pipelined lookahead, speculative burst 게이트, OpenAI 형태 3개 엔드포인트의 요청 배선, 분산 와이어 구조체(`#[serde(default)]` 하위 호환), CLI 플래그 `--top-n-sigma`(별칭 `--top-nsigma`), `docs/python-client.md`.

검증: 코어 sampling 테스트 77개와 루트 크레이트 필터 스위트 green, clippy `-D warnings`와 `cargo fmt --check` green. Qwen3-4B-4bit 실체크포인트에서 온도 2.0에 필터를 켜면 유창한 출력, 끄면 붕괴를 확인했고, greedy 출력은 플래그 유무와 무관하게 토큰 동일했다.

## 4. 후속 조치

- #1377이 훅 위에 `typical_p`를, #1373이 `p_less`를 추가한다. 순서 슬롯은 `apply_row_filters`에 고정되어 있다.
- #1436이 네이티브 `/completion` 필드, `--top-nsigma` 서버 플래그 표면, 매니페스트 엔트리 2건의 플립을 맡는다.
- 드래프터 내부의 greedy `fused_sample` 호출 지점(DFlash 라운드 루프, MTP 헤드)은 설계상 모든 샘플러를 우회한다. 수락률에만 영향을 주는 기존 간극으로 기록해 둔다.
