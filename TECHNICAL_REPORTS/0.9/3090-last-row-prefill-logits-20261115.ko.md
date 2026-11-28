# 기술 보고서: PR #3090 - 모든 모델에서 prefill의 샘플링 행만 projection

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 높음 (218개 파일, 모든 autoregressive 모델)

## 요약

대부분의 모델은 prefill에서 trait 기본 구현을 썼다. forward 전체를 돌리고, `[B, L, vocab]` logits에서 행 하나를 잘라 쓰는 방식이다. final softcap이나 scale이 있으면 같은 크기의 사본이 하나 더 생겼다. 새로 추가한 opt-in hook `LanguageModel::last_row_logits`를 구현한 모델은 샘플링할 행만 projection하고, `forward_last_logits*` 기본 구현 세 개는 모델이 이 hook을 제공하면 그것을 쓴다. 이제 `src/models`와 `src/vision`의 모든 autoregressive `LanguageModel`이 text 경로에서, 그리고 embeddings 경로가 있으면 그 경로에서도 이 hook을 쓴다. VLM wrapper와 Qwen-VL의 mrope/deepstack 경로도 포함된다. M5 Max에서 28K 토큰 기준 단일 패스 피크는 gemma-3n-e2b가 38.17 GB에서 23.09 GB로, qwen2.5-vl-3b가 25.13 GB에서 16.74 GB로, gpt-oss-20b가 30.61 GB에서 22.55 GB로, exaone-4.0-1.2b가 15.70 GB에서 9.77 GB로 줄었다. greedy 출력은 측정한 10개 셀 모두에서 바이트 단위로 같다(#3070).

## 1. 문제 정의

engine prefill은 `try_forward_last_logits_with_sequence_id`(text) 또는 `forward_last_logits_with_embeddings_and_sequence_id`(VLM)를 호출한다. 이를 override한 계열은 열두 개 정도뿐이었다. 나머지 모델은 LM head가 모든 위치를 거쳤고, vocab이 250K이면 기본 2048 토큰 piece마다 약 1 GB, 단일 패스에서는 수십 GB가 들었다. VLM wrapper는 안쪽 text 모델에 override가 있어도 기본 구현으로 빠졌다.

## 3. 기술적 선택과 그 이유

- **hook은 하나다.** 모델은 override 세 개 대신 `last_row_logits` 하나만 구현한다. 기본 구현은 아무것도 하지 않으므로, opt-in하지 않은 모델의 동작은 바뀌지 않는다. `select_row`는 기존 forward 본문에 선택적인 행 위치를 넘기고, `last_row_logits_from`은 위임만 하는 wrapper가 쓴다. `&M`, `LoadedModel`, `VisionLanguageModel`은 hook을 그대로 전달한다.
- **forward 전체의 연산 순서는 그대로다.** 층, recurrent state, KV 쓰기, mrope, deepstack은 여전히 모든 행에서 돈다. final norm, head, softcap이나 scale만 샘플링할 행에서 돈다. 따라서 출력이 바뀌지 않고, logit trace도 필요 없었다.
- **roster guard로 적용 범위가 어긋나지 않게 한다.** `last_row_logits_coverage_tests.rs`는 모든 `impl LanguageModel`을 훑어, hook을 쓰지도 않고 목록에도 없는 구현이 있으면 실패한다. 의도적으로 목록에 올린 것은 네 개다. DiffusionGemma와 Llada2Moe는 denoising loop가 모든 위치를 읽고, Florence2와 NemotronParse는 encoder-decoder runtime이다.

## 7. 변경 요약

| 모델 | 모드 | 피크, main | 피크, 브랜치 | 출력 |
|---|---|---|---|---|
| exaone-4.0-1.2b-4bit | 2048 / 단일 | 5.98 / 15.70 GB | 5.10 / 9.77 GB | 동일 |
| gemma-3n-e2b-it-4bit | 2048 / 단일 | 9.75 / 38.17 GB | 7.67 / 23.09 GB | 동일 |
| gpt-oss-20b-mxfp4-q4 | 2048 / 단일 | 15.88 / 30.61 GB | 13.79 / 22.55 GB | 동일 |
| qwen2.5-vl-3b-instruct-4bit (text) | 2048 / 단일 | 6.55 / 25.13 GB | 5.00 / 16.74 GB | 동일 |
| kimi-vl-a3b-thinking-4bit (text) | 2048 / 단일 | 14.43 / 32.04 GB | 12.84 / 25.57 GB | 동일 |

테스트:
- last-row 테스트 161개: 변환한 아키텍처마다 작은 random-weight 케이스 하나, VLM wrapper와 Qwen-VL / GLM-4V image-state 케이스, roster 테스트로 이뤄진다.
- 전체 `mlxcel` lib suite가 직렬 실행에서 통과한다(11,949개).

자세한 내용: `docs/benchmark_results/last-row-prefill-logits-m5max-2026-11-15.md`.

## 8. 후속 조치

- VLM wrapper 십여 개는 작은 vision fixture가 없어 unit 케이스가 없다. 이들의 hook은 테스트된 text 경로를 얇게 호출할 뿐이다.
- EXAONE-MoE에서 sliding window 이상 위치에서 시작하는 청크 continuation이 dense-cache 경로에서 MLX 오류(`broadcast_shapes`)로 중단된다. 작업 중에 발견했고 main에서는 아직 확인하지 않았다.
