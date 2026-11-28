# PR #3415: speculative exactness 조사와 Inkling verify 블록의 바이트 일치

**날짜**: 2026-11-25
**상태**: Draft PR (closes #3409). M3 Ultra(512 GB)에서 로컬 체크포인트 21개로 검증. 좁은 범위의 lib 테스트, clippy, contract check 통과

## 요약

PR #3405는 Inkling 2의 verify 블록을 decode와 비트 단위로 같게 만들었고, 다섯 원인 중 둘은 8토큰짜리 시작 probe로는 볼 수 없다는 점을 확인했다. 이 PR은 다른 drafter 지원 계열에 같은 원인이 있는지 조사하고, 일회성이던 진단 두 개를 공용 helper로 합치고, 원인이 dense gemv/gemm인 계열 하나(Inkling)를 고치고, sliding window가 있는 계열의 공용 probe에 long-context draw를 추가한다.

조사 결과는 대체로 문제가 없다. exactness gate가 있는 양자화 계열은 모두 narrow 양자화 커널에서 폭 2, 4, 8, 짧은 프롬프트와 600, 2100토큰 뒤 모두 일치한다. Inkling도 이제 일치한다. Gemma 4 26B A4B는 MoE feed-forward에서 여전히 decline된다. gate가 없는 token-only verify round(prompt lookup)는 문서에 적힌 대로 받는 모든 모델에서 불일치한다.

## 1. 조사

방법, 표, 제외한 체크포인트는 `docs/benchmark_results/speculative-exactness-survey-m3ultra-2026-11-25.md`에 있다. 핵심 판정:

| 계열 | 체크포인트 | 판정 |
|---|---|---|
| Qwen 3.5 (DFlash, MTP) | qwen3.5-4b-4bit, qwen3.8-27b-4bit | narrow 커널에서 모든 폭과 길이에서 일치 |
| qwen4_exp (MTP) | Qwen3.8-Flash-Next-4bit-mtp | 동일 |
| Gemma 4 (MTP) | 12B 4-bit, 31B 4-bit | 동일 |
| Gemma 4 MoE (MTP) | 26B A4B 4-bit | 두 커널 모두 decline, narrow에서는 layer 0 MLP 출력 |
| GLM-4.7-Flash (MTP) | glm-4.7-flash-4bit | narrow 커널에서 일치 |
| LFM2 (DSpark) | d1-3b-mlx-8bit | narrow 커널에서 일치 |
| Inkling (MTP) | inkling-0.6b bf16 | 이전에는 layer 0에서 decline, 이후 기본 커널에서 일치 |
| Token-only round | DeepSeek V3 4종, xing4_0 2종, arcee, k2_horizon, OLMoE, GLM | 모든 곳에서 row 0부터 불일치, gate 없음 |

두 가지를 알 수 있다. dense gemv/gemm 원인은 projection이 dense인 곳에서만 나타나고, gate가 있는 bf16 체크포인트는 Inkling 2를 빼면 디스크에 Inkling 0.6B뿐이다. 그리고 gate가 있는 계열에는 long-context 원인이 없다. 8토큰에서 통과한 계열은 600과 2100에서도 통과했다. 오프라인 round loop가 있는 MTP 조합 넷은 2048토큰 prefill chunk를 넘는 프롬프트에서도 drafter 사용 출력이 classic 출력과 같았다.

## 2. 공용 진단

`models::speculative_exactness_diag`는 Gemma 4 probe trace를 옮기고 넓힌 것이다.

- `capture` / `with_capture`: thread-local sink. scope 밖의 `capture` 호출은 thread-local 읽기 한 번이라 서빙 forward에 남겨 둘 수 있다.
- `Stage::of`: forward가 이미 돌려주는 텐서로 같은 기록을 만든다. 레이어별 hidden state를 돌려주는 DFlash target용이다.
- `locate`: 처음 다른 stage, 그 안에서 처음 다른 row, 바이트 수. 형식이 맞지 않는 trace는 오류이며 일치로 읽히지 않는다.
- `report`: stage마다 한 줄, row별 다른 바이트 수.

사용처: Gemma 4 시작 probe(동작 동일), Inkling 시작 probe(신규: decline 사유에 레이어가 나온다), DeepSeek V3 레이어 루프, 조사 테스트, Inkling 2 per-visit 테스트.

## 3. Inkling

진단은 첫 차이를 layer 0 출력, row 0에 놓았다. 원인은 넷이다.

| 원인 | 수정 |
|---|---|
| 모든 dense 곱에서 한 행은 gemv, 여러 행은 gemm | 모델 scope 안에서 `row_invariant::linear` / `dense` |
| short convolution: 한 토큰은 window와 tap의 곱, 여러 토큰은 `conv1d` | verify 블록 안에서는 위치마다 한 토큰 형태 |
| routed expert가 토큰 수에 따라 커널을 바꿈 | MoE 전체에 `row_invariant::per_position` |
| window를 넘으면 블록 row r이 mask된 오래된 key r개를 더 봄 | row별로 자기 window 시작부터 attention |

attention은 `mlxcel_core::inkling_layer`에 있고 Inkling MTP drafter도 이를 쓴다. helper는 scope 밖에서 일반 `matmul`이므로 drafter는 바뀌지 않는다. Inkling의 verify는 speculative append scope에 들어가지 않으므로, verify forward는 명시적 scope(`verify_block_scope`)로 블록을 표시한다.

커널을 쓴 decode는 244.4 tok/s, MLX `matmul`은 239.3 tok/s다(ABAB 8쌍, 6쌍에서 더 빠르고 2쌍은 1% 이내). 그래서 기본으로 켜 두었고 `MLXCEL_INKLING_ROW_INVARIANT=0`이 A/B 스위치다.

부수 발견: `row_invariant_matvec`는 입력을 `device` 포인터로 받았는데, MLX는 원소가 8개 미만인 배열을 `constant`로 바인딩한다. tiny Inkling 테스트 모델(hidden 4)에서 Metal 라이브러리 빌드가 실패해 프로세스가 중단됐다. 커널이 이제 입력을 인덱스로 읽으며 연산은 그대로다.

## 4. Long-context draw

`models::speculative_exactness_draws`가 모든 계열 probe의 계획을 갖는다. 짧은 draw들 뒤에, 계열이 window를 선언하면 `sliding_window + 32`토큰짜리 draw 하나를 더한다. Inkling, Laguna, Muse Glimmer가 선언하고, Gemma 4는 기존 draw를 공용 계획을 통해 유지한다. window가 없는 계열은 이전과 같은 계획을 받으므로 시작 비용이 그대로이며, forward 수를 세는 테스트가 이를 고정한다. `MLXCEL_MTP_PROBE_LONG_CONTEXT`는 draw를 끄거나(`0`) 모든 계열에 길이를 강제한다.

## 5. 테스트

- `a_seeded_mismatch_is_reported_at_its_first_layer_and_row`: 원소 하나를 뒤집은 toy forward에서 helper 검증.
- `the_long_draw_catches_the_sliding_window_case`: Inkling 2의 결함을 가진 가상 sliding 레이어가 짧은 draw는 통과하고 긴 draw는 row 1에서 실패.
- `a_family_without_a_boundary_pays_nothing_for_the_long_draw`.
- `the_verify_block_matches_the_chain_on_metal` (Inkling tiny 모델, head 폭 64): 변경이 있으면 일치, 없으면 layer 0부터 다름.
- `real_checkpoint_probe_fails_without_row_invariant_and_passes_with_it` (ignored, 실제 체크포인트).
- `a_tiny_row_or_weight_does_not_break_the_kernel_build` (core).

## 6. 한계

- Inkling의 MTP 처리량은 측정하지 못했다. 공개된 drafter는 Inkling-Small과만 짝이 맞고, 이번 런에서는 그 체크포인트를 로드할 수 없었다. verify round 비용은 불일치하던 때보다 늘었고, #1549는 Inkling-Small의 MTP가 이미 classic decode보다 느리다고 측정했다.
- 양자화된 Inkling-Small 체크포인트는 probe하지 않았다.
- Laguna와 Muse Glimmer의 긴 draw는 tiny 모델 테스트로만 확인했다.
- Inkling 2의 probe는 여전히 짧은 프롬프트만 쓴다. 그 코드는 #3410 담당이다.
- Gemma 4 26B A4B: 어느 MoE 곱이 다른지는 추적하지 않았다.
