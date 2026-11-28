# PR #2472: sink attention의 Used by 주석 바로잡기

**작성일**: 2026-10-20
**상태**: 구현 완료, PR 열림
**위험도**: Low (cxx bridge의 주석 여섯 줄, 코드 변경 없음)

## 요약

#1786을 닫는다. `fast_scaled_dot_product_attention_with_sinks`는 세 곳에 선언되어 있고(`src/lib/mlxcel-core/src/lib.rs`의 `cxx::bridge` 블록, `mlx_cxx_bridge.h`, `mlx_cxx_bridge.cpp`), 각 선언에 `Used by: GptOss`가 붙어 있었다. 실제 호출 family는 다섯이다. GPT-OSS, Laguna, DeepSeek-V4, Falcon-OCR, 그리고 #2460 이후의 OpenAI Privacy Filter. 같은 주석은 sink를 "per-head attention bias for first position"이라고 설명했는데, sink는 그런 것이 아니다. 이제 세 주석 모두 sink를 올바르게 설명하고 호출 family 다섯을 적는다.

## 문제 정의

`docs/code-guidelines.md`("Shared Function Comments")는 `// Used by:` 줄을 공유 함수를 바꿀 때 어떤 family가 깨질 수 있는지 적어 두는 기록으로 삼는다. 호출자 다섯 중 넷이 빠져 있었으니, sink SDPA 경로를 바꾸면 GPT-OSS만 놓고 검증했을 것이다. 설명도 산술을 오해하게 만드는 방향으로 틀려 있었다. sink는 첫 위치 score에 더하는 bias가 아니다. softmax 분모에 합류하는 head별 추가 logit 하나이며 value 행이 없어서, value 벡터를 보태지 않은 채 확률 질량을 흡수한다. MLX의 fallback은 이를 score 앞에 이어 붙이고 softmax를 적용한 뒤 잘라낸다.

## 기술적 결정

- 설명과 호출자 목록은 이슈의 문구를 쓰고, 이슈가 제기된 뒤 호출자가 된 `OpenAIPrivacyFilter`(#2460)를 더한다. 표기는 `src/models/gpt_oss_common.rs`에 이미 있는 `Used by:` 줄을 따라, 이 family가 어디서나 같은 이름으로 불리게 한다.
- GPT-OSS SwiGLU activation의 `Used by: GptOss` 세 줄은 이슈가 요구한 대로 그대로 둔다. 이 함수는 `mlxcel-core` 수준에서는 GPT-OSS만 쓴다(privacy filter는 clipped SwiGLU를 `gpt_oss_common.rs`를 통해 공유하며, 그 파일에 별도의 `Used by:` 줄이 있다).
- 주석만 바꾼다. 빌드는 돌리지 않았다. C++ 수정은 `//` 주석 안이고 Rust 수정은 bridge 블록 안의 `///` doc 주석이다.

## 변경 사항

- `src/lib/mlxcel-core/src/lib.rs`, `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h`, `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`: 각 지점에서 `fast_scaled_dot_product_attention_with_sinks` 위의 주석 두 줄 교체.

## 검증

- `cargo fmt --all -- --check` 통과.
- `grep -rn 'fast_scaled_dot_product_attention_with_sinks(' src/models`는 `gpt_oss.rs`, `laguna_layers.rs`, `deepseek_v4_attention.rs`, `falcon_ocr.rs`, `openai_privacy_filter_layers.rs`, `openai_privacy_filter_tests.rs`를 나열한다. `examples/`, `tests/`, `benches/`에는 호출 지점이 없다.
- `git diff -U0`는 정확히 여섯 줄, 모두 주석만 바뀌었음을 보여 준다.

## 후속 조치

- 호출자 목록은 이슈 제기와 이 수정 사이에 한 번 더 어긋났다(#2460이 호출자를 추가). #1141은 `// Used by:` 관례를 staleness까지 포함해 CI에서 강제하자고 제안한다. 이 PR은 그런 검사가 잡아낼 종류의 drift다.

## 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 3 |
| 추가된 라인 | 6 |
| 삭제된 라인 | 6 |
| 테스트 추가 | 0 |

## 관련

- Issue #1786: chore(core): correct the sink attention Used by comment (PR #1779 리뷰 중 발견).
- PR #2460: 다섯 번째 호출자인 OpenAI Privacy Filter.
- Issue #1141: chore: `// Used by:` 관례를 staleness까지 CI에서 강제.
