# PR #3423: 1세대 Inkling의 prompt lookup 검증 라운드

**날짜**: 2026-11-25
**상태**: Draft PR 열림 (#3422 종료). M3 Ultra (512 GB)에서 `inkling-0.6b-a0.6b`와 `inkling-small-mlx-4bit`로 검증했습니다. 좁은 범위의 lib 테스트, clippy, 계약 검사를 통과했습니다.

## 요약

`--prompt-lookup`과 서버의 토큰 전용 검증 라운드는 1세대 Inkling을 거부했습니다. 캐시를 모델이 소유하고, padded prefill을 지원하지 않으며, 검증 블록을 되감을 수 있다고 선언하지 않았기 때문입니다. 정확한 되감기는 PR #3419가 MTP용으로 이미 만들어 두었습니다. 검증 스코프 안에서 각 짧은 컨볼루션이 입력 창 전체를 기록하고, `keep_verified_prefix`가 그 창에서 수락된 접두사의 상태를 다시 만듭니다. 그러나 토큰 전용 라운드는 이 스코프에 들어가지 않아서 아무것도 기록되지 않았고, 모델에는 `rewind_decode_appends`도 없었습니다.

이 PR은 둘을 연결합니다. 온도 0의 prompt lookup 출력은 두 체크포인트 모두에서 일반 디코드와 같습니다. 모델별 서베이의 engine 경로도 시도한 모든 폭과 프롬프트 길이에서 equal입니다. 복사가 많은 응답의 속도는 Inkling-Small에서 일반 디코드의 1.14배, 0.6B에서 2.12배입니다.

## 1. 동작 방식

- **스코프.** 행 커널이 켜져 있고 `row_invariant::is_verify_block(shape)`가 참이면 `forward_with_sequence_id`가 `verify_block_scope()`에 들어갑니다. 이 조건은 엔진의 `DecodeLookaheadAppendScope` 안에서 실행되는, 한 시퀀스의 2~16행짜리 추측 추가를 뜻합니다. 스코프 안에서는 dense 곱, 컨볼루션, 전문가, 어텐션이 #3409와 #3280의 행 불변 형태로 실행되므로 각 행이 디코드 스텝과 같습니다. 각 컨볼루션은 `[1, kernel - 1 + L, C]` 창도 저장합니다. 일반 prefill은 추측 추가가 아니므로 바뀌지 않습니다.
- **게이트.** `supports_speculative_verify_rewind`는 `row_kernel_enabled()`를 반환합니다. 그러면 `prompt_lookup_unsupported_reason`과 `Engine::verify_rounds_unsupported`가 기존의 `model_rewinds` 분기로 이 모델군을 받아들입니다. `prompt_lookup.rs`의 공용 게이트는 손대지 않았습니다. `MLXCEL_INKLING_ROW_INVARIANT=0`이면 블록이 정확하지도 않고 기록되지도 않으므로 게이트는 계속 거부합니다.
- **되감기.** `rewind_decode_appends(seq, n)`은 새 함수 `inkling_layer::drop_newest_verify_rows`를 호출합니다. 캐시는 커널 크기를 모르지만, 순전파가 남긴 상태는 `kernel - 1`행입니다. 따라서 블록 길이는 창의 행 수에서 상태의 행 수를 뺀 값입니다. 이 함수는 먼저 모든 레이어에 창이 있는지, 모든 레이어의 블록 길이가 같은지, `n`이 그 길이를 넘지 않는지 확인합니다. 그런 다음에야 각 레이어에서 `keep_verified_prefix(appended, appended - n)`을 호출합니다. 이제 `keep_verified_prefix`는 `kept == 0`을 받아들입니다. 이 경우 KV 꼬리는 블록 전체만큼 잘리고, 각 컨볼루션 상태는 창의 처음 `kernel - 1`행, 즉 블록 이전의 상태가 됩니다. `failed_round`와 `warm_up_verify_widths`가 추가분을 모두 되돌리므로 엔진에는 이 경우가 필요합니다. MTP는 항상 1행 이상을 남기므로 동작이 바뀌지 않습니다.
- **넓은 블록.** `--prompt-lookup-max-draft`는 제안을 최대 64개까지 허용하므로 블록은 최대 65행이 됩니다. 이제 16행보다 넓은 추측 추가도 행 불변 스코프 밖에서 실행되지만 창을 기록합니다. 그래서 그런 블록의 거부된 꼬리도 올바르게 되감깁니다. 다만 블록 자체는 체인과 바이트 단위로 같지 않습니다. 이 기록이 없으면 그렇게 넓은 블록은 되감기에 실패하고 요청이 끝납니다.
- **연결.** 공개 체크포인트의 래퍼인 `InklingVlModel`이 두 훅을 위임합니다. CLI와 서버 스케줄러는 모두 `Engine::verify_round`를 거치므로 따로 코드가 필요하지 않았습니다. `supports_padded_prefill`은 `false`로 두었고, `trim_state`는 계속 컨볼루션 상태를 비웁니다. 그 주석은 이제 padded prefill 전용이라고 설명하며, #3419 보고서 66행도 더는 이 훅을 prompt lookup의 되감기라고 부르지 않습니다.

## 2. 정확성

`survey_block_vs_chain`의 engine 경로는 모델 소유 레이아웃에서 항상 `NotRun`을 반환했습니다. 이제 `LoadedModel::Inkling`과 `InklingVLM`에서도 실행됩니다. 이 둘의 `make_caches`는 두 측정 갈래가 함께 쓰는 기본 상태를 초기화합니다. Inkling 2는 계속 제외되므로 판정이 바뀌지 않습니다.

| 체크포인트 | 폭 | 프롬프트 | engine 판정 |
|---|---|---|---|
| inkling-0.6b-a0.6b | 2, 4, 8 | 8, 600, 2100 | equal (9/9) |
| inkling-small-mlx-4bit | 2, 4, 8, 16 | 8, 600, 2100 | equal (12/12), wide 양자화 커널 |

## 3. 처리량

`mlxcel generate --temp 0`을 GPU 잠금 아래에서 ABAB 4사이클로 측정했습니다. 측정 중에도 다른 워크트리가 빌드하고 있었습니다. tok/s에는 prefill이 포함됩니다. 모든 쌍의 출력 텍스트가 같았습니다.

| 체크포인트, 프롬프트 | 일반 디코드 | prompt lookup | 비율 | 순전파당 토큰 |
|---|---|---|---|---|
| Small, 복사 (550 토큰) | 42.12 | 48.00 | 1.14x | 2.26 |
| Small, 복사, 400 토큰에서 끊음 (추론만) | 42.29 | 44.00 | 1.04x | 1.89 |
| 0.6B, 반복 문단 (50 토큰) | 235.52 | 500.41 | 2.12x | 6.13 |
| 0.6B, 초안을 만들 것이 없는 응답 (52 토큰) | 234.24 | 190.85 | 0.81x | 0.98 |

Small에서 비율이 순전파당 토큰보다 훨씬 낮은 것은 정확한 검증 블록이 전문가와 어텐션을 위치마다 따로 실행하기 때문입니다. 블록당 비용은 따로 측정하지 않았습니다.

## 4. 테스트

- `tiny_tests::prompt_lookup_gate_follows_the_row_kernel`: 행 커널이 켜져 있으면 게이트가 `None`을, 꺼져 있으면 `Some`을 반환합니다. 기본 `supports_speculative_verify_rewind`가 `false`인 이전 코드에서는 실패합니다.
- `prompt_lookup_tests::a_rewound_verify_round_matches_the_single_token_chain_on_metal`: 폭 4와 10에서 `kept`를 0부터 `width - 1`까지 모두 확인합니다. 각 경우 검증 블록을 추측 추가로 실행하고, `rewind_decode_appends`로 되감은 뒤 디코드 스텝을 두 번 실행합니다. 이 로짓은 남은 토큰을 하나씩 디코드한 모델의 로짓과 바이트 단위로 같아야 합니다.
- `a_rewind_without_a_recorded_block_is_refused_without_mutation`: 일반 스텝 뒤, 블록 길이를 넘는 `n`, 두 번째 되감기, 알 수 없는 시퀀스에서는 되감기가 거부됩니다. 이후에도 상태는 체인과 같습니다.
- `a_verify_block_past_the_row_invariant_scope_still_rewinds`: 20행 블록에서 15행을 버린 뒤 체인과 2e-4 이내로 일치하는지 확인합니다.

## 5. 하지 않은 것

- 서버 경로는 끝에서 끝까지 실행해 보지 않았습니다. 서버는 CLI와 같은 게이트와 라운드 코드를 씁니다.
- 0.6B 체크포인트는 코드를 이어 쓰지 않으므로 지시형 복사 프롬프트에서는 초안이 생기지 않습니다. 그래서 이 모델의 처리량 행에는 응답이 프롬프트를 반복하는 프롬프트를 썼습니다.
