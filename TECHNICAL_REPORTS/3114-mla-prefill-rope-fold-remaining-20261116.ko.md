# 기술 보고서: PR #3114 - 남은 수동 mask MLA prefill을 SDPA로 합침

**작성일**: 2026-11-16
**상태**: 완료 (KimiLinear, LongcatFlashNgram, GLM4MoeLite pipeline stage는 unit test로만 확인)
**위험도**: 중간

## 요약

#3083에서 추가한 `mla::rope_folded_attention`은 rope 부분을 SDPA에 이어 붙여, prefill에 head별 additive score 텐서도 dense mask도 필요 없게 한다. 그런데 glm4_moe_lite, kimi_linear의 MLA 층, kimi_k3, longcat_flash_ngram은 여전히 causal mask를 직접 더하고 있었다. 이제 이들의 다중 토큰 단계는 absorbed와 materializing 두 분기 모두 MLX causal mode로 이 함수를 거치고, GLM4MoeLite pipeline stage도 같은 방식으로 mask를 쓰지 않는다(#3099).

M5 Max에서 glm-4.7-flash-4bit(22,754 토큰)의 2048 토큰 청크 피크가 28.82 GB에서 20.51 GB로 줄었고, prefill은 35~50초에서 12.3~12.6초로 짧아졌다. 단일 패스는 이전에 82.8 GB에서 Metal GPU timeout으로 실패했지만, 이제 24.11 GB로 끝난다.

## 3. 기술적 선택과 그 이유

- **decode와 호출자 mask.** 단일 토큰 decode는 그대로다. 호출자가 넘긴 mask도 그대로 적용하므로, GLM4MoeLite MTP drafter는 자기 mask를 계속 쓰고 `create_causal_mask` 호출 목록에 남는다. 목록 문장은 이제 17개 파일로 적혀 있다.
- **연산 순서가 바뀐다.** rope를 이어 붙이면 score를 더하는 순서가 달라진다. glm-4.7-flash의 teacher-forced trace에서 decided position 불일치는 width 2048 cold에서 3,335개 중 2개(0.060%), 앞에 2048 토큰 문맥이 있는 width 4096에서 3,360개 중 3개(0.089%)다. perplexity는 각각 +0.23%, +0.05% 움직였다. 불일치의 81~83%는 브랜치의 2순위가 main의 1순위였다. 2048 청크 greedy 출력은 바이트 단위로 같다.

## 7. 변경 요약

| 모델 | prefill | 피크, main → 브랜치 | prefill, main → 브랜치 |
|---|---|---|---|
| glm-4.7-flash-4bit | 2048 청크 | 28.82 → 20.51 GB | 35~50 → 12.3~12.6 s |
| glm-4.7-flash-4bit | 단일 패스 | 82.8 GB에서 GPU timeout → 24.11 GB | 실패 → 11~16 s |
| kimi-k3-2l-mla-synth (2층, 2,831 토큰) | 단일 / 2048 | 41.51 → 39.69 GB | 0.50 → 0.45 s |

테스트:
- 계열마다 작은 가중치로 접힌 prefill 결과를 단일 토큰 경로와 비교하는 테스트를 새로 넣었고, pipeline stage는 단일 프로세스 모델과 비교한다.
- `mlxcel-core` lib suite가 통과한다(2,318개).
- 전체 `mlxcel` lib suite는 `server::chat_template_json::tests::non_string_object_keys_are_coerced_like_python` 하나를 빼고 통과한다. 이 테스트는 #2819 이후 main에서도 똑같이 실패하며, #3115에서 고쳤다.
