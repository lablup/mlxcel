# 기술 보고서: 이슈 #1373 - 하이퍼파라미터 없는 p-less 샘플링

**Date**: 2026-09-29

**Status**: Apple Silicon에서 로컬 구현 및 검증 완료, 머지 대기.

**Languages**: Rust

**Risk Level**: Low (기본 비활성, 꺼져 있으면 토큰 스트림이 바이트 단위로 동일)

## 요약

mlxcel에 `p_less` 샘플러를 추가한다. 불리언 요청 필드와 `mlxcel generate`, `mlxcel chat`의 `--p-less` 플래그로 제공된다. 필터는 온도로 스케일한 softmax에서 확률이 행의 충돌 확률 `L = sum_v p(v)^2` 이상인 토큰만 남기고 나머지를 `-inf`로 마스킹한다. 조정할 파라미터가 없다. #1375가 도입한 행 필터 훅의 세 번째 필터로 구현되어 새 디스패치 경로가 필요 없었다.

## 1. 문제 정의

top-k, top-p, min-p, typical-p, top-n-sigma 같은 절단 샘플러는 모두 모델과 온도마다 사용자가 조정해야 하는 임계값이 필요하다. p-less는 분포 자체에서 임계값을 도출한다. `L <= max_v p(v)`이므로 argmax는 항상 살아남고, 뾰족한 행은 argmax 하나로 수렴하며, 평평한 행은 어휘의 대부분을 유지한다. mlxcel에는 이 샘플러가 없었다.

## 2. 변경 요약

- `SamplingConfig::p_less: bool`(기본 `false`)과 `SamplingConfig::effective_p_less()`. 그리디 경로와 mirostat에서는 `false`로 접힌다.
- `sampling.rs`: `p_less_filter(logits, temperature)`, `apply_row_filters`의 `top_n_sigma`와 `typical_p` 사이 `p_less` 분기, `apply_extended_chain`의 동일 필터, `FusedSampleParams::p_less`(`matches`에서 비교하므로 균일한 배치는 단일 디스패치 유지).
- #1375 패턴을 따른 요청 배선: `SamplingParams::p_less`, `RequestOptionOverrides::p_less`, 두 `build_sampling_config` 분기를 지나는 `ResolvedSamplingParams::p_less`, `NativeCompletionRequest::p_less`, 분리형 서빙의 `SerializableSamplingState::p_less`(`#[serde(default)]`), 배치 추측 윈도우 동등성 비교.
- CLI: `SamplingOptions`의 `--p-less`, `generate`와 `chat`이 사용.
- 문서: `docs/server-features.md`, `CHANGELOG.md`.

## 3. 기술적 결정

**필터가 온도를 명시적으로 받는다.** 융합 C++ 체인은 행 필터 이후에 온도를 적용하므로, 그대로면 필터는 온도가 적용되지 않은 분포를 본다. p-less는 온도가 적용된 분포에서 정의되므로 `p_less_filter`가 직접 `T`로 나눈다(`T == 1.0`이면 생략). 융합 체인 자체의 스케일링은 그대로다. 따라서 온도를 올려도 유지 집합이 줄어들지 않는다.

**마스킹된 항목은 특별 처리 없이 유지된다.** `-inf` 로짓의 확률은 정확히 0이고, 유한한 항목이 있는 행에서 `L > 0`이므로 `0 >= L`은 거짓이다. 그래서 기존 마스크가 유지되고, 뒤에 `-inf`를 덧붙여도 `L`과 유지 집합이 변하지 않는다.

**원시 필드가 아니라 유효값을 쓴다.** `FusedSampleParams::from_config`는 `effective_p_less()`를 저장하므로, 의미 없는 `p_less` 차이만 있는 그리디 행 때문에 융합 배치나 추측 윈도우가 갈라지지 않는다. `effective_top_n_sigma`와 같은 방식이다.

**확장 체인.** `apply_extended_chain`은 `apply_row_filters`를 거치지 않으므로 같은 위치에서 필터를 따로 적용한다. 어느 경로에서도 두 번 적용되지 않는다.

**서버 전역 기본값 없음.** 불리언에는 정리할 비활성 센티널이 없고, 이슈가 필드 누락 시 `false`로 해석하도록 정하므로 서버 `--p-less` 플래그와 런타임 설정은 두지 않았다.

## 4. 검증

- `sampling.rs`의 단위 테스트(무작위 40개 행에 대한 호스트 참조 일치, argmax 생존, 뾰족한 행, 온도 단조성, 행 독립성, `-inf` 패딩, 그리디 생략, 융합 파라미터 동등성, 128행 `batched_fused_sample`)와 요청, 실행, 와이어, 추측 윈도우 테스트 모듈.
- Qwen3-4B-4bit 실제 체크포인트 검증은 PR 본문에 기록한다.

## 5. 미수행

서버 전역 기본값과 `/props`, 네이티브 `generation_settings`에서의 `p_less` 에코는 이슈가 요구하지 않아 포함하지 않았다.
