# 기술 보고서: PR #1094 - Gemma 4 서버 thought 마커 제거

**작성일**: 2026-08-10
**상태**: 완료
**언어**: Rust
**위험도**: Medium

## 요약

PR #1094는 최종 답변 내용과 streaming/non-streaming 동등성을 유지하면서 Gemma 4의 `thought` channel 인자가 서버 `reasoning_content`에 남는 문제를 해결한다. 리뷰 과정에서는 더 긴 `<|channel>thought` delimiter를 소비하면서 드러난 token 위치 계산과 비정상 stream 처리 문제 두 가지도 함께 교정했다.

## 1. 문제 정의

기존 서버 stream filter는 `<|channel>`만 소비한 뒤 reasoning 상태로 전환했기 때문에 뒤따르는 `thought` 인자가 reasoning text로 노출되었다. Delimiter를 단순히 `<|channel>thought`로 바꾸면 인접한 두 위험이 생긴다. Delimiter byte와 출력 text에 걸친 decoded token이 병렬 logprob queue에서 두 번 계산될 수 있고, 올바른 긴 opener를 선점하지 않으면서도 비정상 bare `<|channel>`을 계속 억제해야 한다.

## 2. 기술적 선택과 그 이유

### 2.1 전체 opener를 우선하되 fallback을 지연 유지

Delimiter table에는 `<|channel>thought`와 `<|channel>`을 모두 유지한다. 다만 현재 buffer가 긴 opener로 완성될 가능성이 있는 동안에는 짧은 match를 지연한다. End-of-stream에서는 추가 fragment가 올 수 없으므로 모호성을 해소하여 잘린 channel markup이 `flush()`를 통해 노출되지 않게 한다.

### 2.2 Fragment 위치가 이미 계산되었는지 추적

Buffer의 각 decoded fragment는 남은 byte 길이와 token 위치 계산 여부를 담은 span으로 표현한다. Delimiter가 fragment 일부만 소비하면 나머지를 이미 계산된 상태로 표시하여, 이후 reasoning 또는 content 출력 시 동일한 logprob 위치를 두 번 drain하지 않게 한다.

## 3. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 2 |
| 추가 라인 | 167 |
| 삭제 라인 | 33 |
| 주요 모듈 | `server::tool_calls::stream_filter`, `server::routes::chat` |

- 서버 reasoning이 Gemma 4 전체 channel opener를 소비하여 `reasoning_content`에 `thought`를 내보내지 않는다.
- Streaming과 non-streaming 추출이 같은 filter 동작을 재사용하며 최종 visible content는 바뀌지 않는다.
- Fragment-span bookkeeping으로 delimiter가 decoded token 중간에서 끝날 때도 token/logprob 정렬을 보존한다.
- 비정상 출력 호환성을 위한 bare-channel fallback을 유지하고 stream flush 시 안전하게 해소한다.
- Whole/split opener, delimiter 위치 계산, 다른 delimiter family 호환성, end-of-stream fallback을 회귀 테스트로 고정했다.

## 4. 리뷰 발견 사항

| 발견 사항 | 심각도 | 해결 |
|-----------|--------|------|
| 분할된 `thought\n` fragment가 한 token 위치를 두 번 계산할 수 있음 | Medium | 계산 여부를 가진 fragment span으로 수정 |
| Bare channel delimiter 제거 시 비정상 channel text가 노출될 수 있음 | Medium | 짧은 delimiter match 지연으로 수정 |
| 정확히 bare channel에서 잘린 stream이 `flush()` 중 노출될 수 있음 | Low | End-of-stream 최종 delimiter 해소로 수정 |

리뷰 이후 남은 Critical 또는 High 발견 사항은 없다. Delimiter matching 비용은 정적 delimiter table과 최장 delimiter tail 길이로 제한된다.

## 5. 검증

- `cargo test --lib server::tool_calls::stream_filter::tests`: 78 passed, 0 failed.
- `cargo test --lib gemma4`: 225 passed, 0 failed, 기존 hardware-dependent test 30개 ignored.
- `cargo test --lib reasoning_split_identical_whole_vs_chunked`: 통과.
- `cargo test --lib extract_reasoning_gemma4`: 통과.
- `cargo fmt --check`: 통과.
- Hosted PR check 통과. PR이 MLX pin을 변경하지 않아 `MLX pin extraction`은 skip되었다.

## 6. 관련 작업

- Issue #890: Gemma 4 reasoning marker 문제의 서버 측 후속 작업.
- Issue #884: `<|channel>thought` 전체를 하나의 opener로 소비하도록 한 이전 CLI 수정.

