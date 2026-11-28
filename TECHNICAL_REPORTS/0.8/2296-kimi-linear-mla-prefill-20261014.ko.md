# PR #2296: Kimi Linear MLA prefill을 유한하게 복원하기

**날짜**: 2026-10-14

**상태**: 구현 및 독립 리뷰 완료, 집중 테스트와 고정 체크포인트 검증 통과

**위험도**: 중간 (Kimi Linear full-attention 계층의 causal masking 변경)

## 요약

PR #2296은 CUDA에서 Kimi Linear 추론을 복구합니다. 기존 모델은 21토큰 산술 프롬프트와 3,411토큰 검색 프롬프트 모두에서 토큰 ID 0을 반복했습니다. Kimi MLA 경로는 upstream의 불리언 mask 연산을 옮겼지만 mlxcel의 공용 `create_causal_mask`는 가산형 mask를 반환합니다. 0과 음의 무한대를 조건으로 해석하면서 가시성이 뒤집혔고, 모든 prefill의 마지막 query가 완전히 차단된 행을 softmax하게 됐습니다.

이제 MLA 경로는 가산형 mask를 positional score dtype으로 캐스팅한 뒤 더합니다. 회귀 테스트 두 개가 mask 값 자체를 고정하고 실제 3토큰 Kimi MLA prefill을 문제 연산까지 통과시킵니다. 고정한 48B-A3B 4비트 체크포인트는 요구한 chunk 설정에서 `4`와 `AMBER-RAVEN-7429`를 정확히 답합니다.

## 근본 원인

`KimiLinearModel::forward`는 `mlxcel_core::utils::create_causal_mask`로 causal mask를 만듭니다. 이 함수의 계약은 볼 수 있는 key에 0, 차단한 key에 음의 무한대를 놓는 것입니다. `KimiMLAAttention::forward`는 이 배열을 불리언처럼 `where_cond(mask, pe_scores, negative_infinity)`에 넘겼습니다.

0은 false로, 음의 무한대는 true로 변환됩니다. 따라서 볼 수 있는 key는 음의 무한대로 바뀌고 미래 key는 남았습니다. 여러 토큰을 prefill할 때 마지막 query 행에는 미래 key가 없으므로 mask 행 전체가 0이고 positional score 전체가 음의 무한대가 됐습니다. Content score를 더해도 유한한 값이 복구되지 않습니다. 전부 음의 무한대인 행의 softmax가 NaN을 만들고, 첫 MLA 계층부터 이후 residual, recurrent state, 최종 logits로 퍼졌습니다. Greedy argmax는 그 결과 토큰 ID 0을 반환했습니다.

이 결함은 chunking과 무관합니다. 배포된 채팅 템플릿은 예상한 token ID를 만들었고, 체크포인트 packing은 loader와 일치했으며, 이 체크포인트는 두 expert group 값이 모두 1이라 grouped routing 차이도 발화하지 않습니다.

## 변경

`apply_mla_causal_mask`가 저장소의 mask 계약을 보존합니다. Mask dtype이 `pe_scores`와 다를 때만 캐스팅하고 두 배열을 더합니다. Kimi K3와 DeepSeek MLA의 기존 변환 방식과 같고, 캐시를 포함한 `[L, L + offset]` key 축도 일반 broadcast로 처리합니다.

KDA RMS 정규화는 의도적으로 바꾸지 않았습니다. 최신 upstream의 epsilon 식은 별도의 포팅 차이지만 결정적인 ID 0 반복을 설명하지 않습니다. Mask 하나만 고친 후보가 체크포인트의 정확 답변을 복구했으므로 관계없는 수치 변경까지 넣으면 입증한 원인이 흐려집니다.

## 회귀 테스트

`kimi_mla_applies_the_repository_causal_mask_as_additive_values`는 알려진 score에 3x3 causal mask를 적용합니다. 결과 행이 `[score, -inf, -inf]`, `[score, score, -inf]`, 세 개의 visible score가 되도록 요구합니다. Mask 반전과 mask 누락을 모두 잡습니다.

`kimi_mla_multi_token_prefill_keeps_the_last_query_row_finite`는 작은 zero-weight `KimiMLAAttention`을 적재하고 3토큰 prefill을 수행해 모든 출력이 유한하도록 요구합니다. 1토큰 fixture는 causal-mask 분기에 들어가지 않으므로 이 결함을 재현할 수 없습니다.

수정 뒤 두 테스트가 통과했습니다. 통제한 음성 대조에서는 helper 하나만 이전 `where_cond` 방식으로 바꾸자 두 테스트가 exit 101로 모두 실패했습니다. 첫 visible score가 0 대신 음의 무한대였고, 실제 MLA fixture의 마지막 행이 유한하지 않았습니다. 그 뒤 수정 소스를 바이트 단위로 복원하고 최종 빌드를 수행했습니다.

## 고정 체크포인트 검증

검증 체크포인트는 `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` revision `919175d75427f1c10cfc17e7254b53295edddf72`입니다. CUDA compute capability 12.1인 NVIDIA GB10에서 실행했고, greedy generation은 temperature 0과 seed 1749를 사용했습니다.

`190b83c3` 기준선의 binary SHA-256은 `592c33a141ccd7a497732e132ac6da302661e22400e27fbcdb2cb3ca858ae8ff`입니다. 장문 프롬프트에서 기본 chunking과 `MLXCEL_PREFILL_CHUNK=0` 모두 생성 토큰 64개가 전부 ID 0이었고, decode 결과는 느낌표 64개였습니다. 21토큰 산술 프롬프트도 토큰 ID 0과 `!`를 냈습니다. PR #2293의 chunk capability 변경 뒤에도 같은 잘못된 값이 나와 no-ID cache reset이 수치 결함의 원인은 아님을 확인했습니다.

첫 mask-only 후보는 base `c9e062af`, source SHA-256 `55c28b20dfd54398e8f6603149e8affbeb6883904c6ee012b277e12d370b429e`, binary SHA-256 `2b2402c0316e9c6d937fa32998c360dba5214dcd7169fc2d9153e6af19472817`을 사용했습니다. 결과는 다음과 같습니다.

- 짧은 프롬프트, 입력 21토큰, `MLXCEL_PREFILL_CHUNK=0`: `4`, token ID `[19]`.
- 장문 프롬프트, 입력 3,411토큰, 기본 chunking: `AMBER-RAVEN-7429`, token ID `[2102, 12169, 16134, 8858, 1254, 12, 51413, 24]`.
- 장문 프롬프트, 입력 3,411토큰, `MLXCEL_PREFILL_CHUNK=0`: 같은 정확 답변과 token ID.

최종 브랜치는 PR #2293이 들어간 main commit `398f9ba8`을 병합해 head `7c3f697706c55d79a54b099b0728892e10ad2115`가 됐습니다. Binary SHA-256은 `b2cb19e07aa833d90e25c039daf13ac43190bf67ce4cf0074c6e074238c40de0`입니다. 요구한 세 실행은 초기 후보와 같은 정확 답변과 token ID를 다시 냈습니다. `MLXCEL_SDPA_DETERMINISTIC=0`으로 실행한 추가 두 건에서도 짧은 입력은 `4`, 기본 장문 입력은 `AMBER-RAVEN-7429`를 그대로 반환해, 수정이 비교 harness의 deterministic SDPA 설정에 기대지 않음을 확인했습니다.

## 검증

- `cargo test --locked -p mlxcel -p mlxcel-surgery --lib --test chat_template_kwargs --profile test-fast --features cuda -j4 --no-run`이 5분 58초에 통과했습니다.
- 최종 집중 회귀는 3개 통과, 0개 실패입니다. 새 MLA mask 테스트 두 개와 PR #2293의 loaded-model chunk-capability 테스트를 포함합니다.
- 공용 이슈 contract는 Python 123개와 WebUI 49개가 exit 0으로 통과했습니다.
- Main 병합 뒤 포맷과 patch whitespace 검사가 통과했습니다.
- 독립 correctness 및 security 리뷰에서 이 범위의 추가 지적이 없었습니다.

## 범위

이 PR은 Kimi Linear MLA masking을 복구하고 #2292를 닫습니다. PR #2293은 Kimi Linear의 model-owned recurrent state가 chunk 사이에서 안전하게 재시작할 수 없어 no-ID chunked prefill을 별도로 비활성화합니다. 최종 브랜치는 두 변경을 모두 포함하지만 테스트와 전후 증거는 원인을 분리해 기록합니다.

## mlxcelverse 및 최신 MLX pin과의 최종 통합

#2304의 MLX `243bdf09` pin, mlxcelverse 경로, rustls 0.23.46을 포함한 main `adf0e589`를 소스 충돌 없이 통합했다. 실제 추론 검증의 staged source tree는 `2e4f5211f42b21973d097a00ed60a7768d7f05ba`이며 CLI SHA-256은 `45753012854036e8b44d1adc622a6eba8bc9d15b601e098d340769581cb1a064`다. 통합 이전 결과를 재사용하지 않고 최종 결합 소스를 새로 빌드했다.

NVIDIA GB10에서 revision `919175d75427f1c10cfc17e7254b53295edddf72`의 `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit`가 정답 비교 5건을 모두 통과했다. 짧은 prompt는 `4`를 반환했고, 3,411-token 긴 prompt는 기본 prefill, 명시적 단일 청크 prefill, 일반 SDPA에서 모두 `AMBER-RAVEN-7429`를 회수했다. 짧은 prompt도 일반 SDPA에서 통과했다. 별도의 일반 SDPA 자연어 검증에서는 낮의 Rayleigh 산란과 일몰의 긴 대기 경로를 설명하는 일관된 두 문장, 58 token을 생성했다. token 0 반복 현상은 없었다. 이 정확성 실행으로 성능을 주장하지 않는다.

최종 소스로 다시 빌드한 `models::kimi_linear` CUDA 테스트 10개가 모두 통과했다. Additive mask, 마지막 query 행의 유한값, 가중치를 적재한 모델의 청크 지원 계약 회귀 테스트를 포함한다. Formatting과 diff 검사도 통과했다.
