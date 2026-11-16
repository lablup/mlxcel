# 기술 보고서: PR #3136 - fused 커널이 도는 곳에서 DeepSeek V2 decode를 기본적으로 absorbed로 실행

**작성일**: 2026-11-16
**상태**: 완료 (CUDA·ROCm은 #3133에서 측정할 때까지 decompressed 기본값 유지)
**위험도**: 중간 (Apple Silicon의 모든 DeepSeek V2 사용자에게 기본 캐시 레이아웃이 바뀜)

## 요약

이제 DeepSeek V2는 #3129의 fused Metal absorbed-decode 커널이 돌 수 있는 곳이면 기본적으로 MLA 잠재값 `(ckv, kpe)`를 캐시하고 absorbed decode를 실행합니다. 그 밖의 환경에서는 head별로 펼친 decompressed K/V 캐시를 그대로 씁니다. `MLXCEL_MLA_ABSORBED`는 세 가지 값을 갖습니다. 설정하지 않으면 fused 커널 사용 가능 여부를 따르고, `=1`은 absorbed를, `=0`은 decompressed를 모든 백엔드에서 강제합니다. 항상 absorbed로 도는 Kimi-VL도 fused 커널로 처음부터 끝까지 검증했습니다. decided 불일치는 0개이고, decode는 composed 경로보다 6.7~7.5% 빠릅니다.

## 1. 문제 정의

#3102는 짧은 문맥에서 absorbed decode가 10% 정도 느리다는 이유로 decompressed 기본값을 유지했습니다. #3129가 양자화 fold와 fused Metal 커널로 그 비용 대부분을 없앴고, 다시 측정해 전환 조건 두 가지가 모두 충족됨을 확인했습니다. 전환 자체는 이번 변경으로 미뤄 두었습니다. fused 커널은 Metal에만 있으므로, 측정 근거도 커널이 실제로 도는 빌드에만 해당합니다.

## 3. 기술적 선택과 그 이유

- **기본값은 플랫폼이 아니라 커널을 따름.** `AbsorbedMode::resolve(fused_available)`는 `mla::fused_decode_enabled()`가 참일 때만 Default를 absorbed로 정합니다. Metal 커널 포트가 있고 `MLXCEL_MLA_FUSED_DECODE`가 꺼져 있지 않아야 합니다. 커널이 없으면 absorbed decode는 composed 경로로 가고, 이 경로에는 #3102가 측정한 짧은 문맥 비용이 아직 남아 있습니다. 그래서 CUDA와 ROCm, 그리고 커널을 끈 Metal은 #3133에서 측정할 때까지 decompressed로 남습니다. 부수 효과로, Metal에서 `MLXCEL_MLA_FUSED_DECODE=0`만 설정해도 V2가 decompressed로 바뀝니다.
- **알 수 없는 값은 Off가 아니라 Default.** `MLXCEL_MLA_FUSED_DECODE`와 같은 정책입니다. 명시적으로 끄는 값을 줘야만 기본 레이아웃을 벗어납니다.
- **greedy parity 테스트를 teacher-forced로 변경.** `tests/deepseek_v2_greedy_parity.rs`는 자유 생성한 greedy id 20개를 mlx-lm reference와 비교했고, 새 기본값에서 1단계부터 실패했습니다. decompressed 레이아웃은 `.`에 29.109, `,`에 29.094점을 줘 f16 한 ULP 차이인데, fused 커널은 둘 다 29.094점을 줍니다. 그래서 argmax가 더 작은 id를 고르고 이후 모든 토큰이 달라집니다. composed absorbed 경로는 reference와 일치했으므로 커널 결함이 아니라 반올림에서 생긴 근소한 동점입니다. 이제 테스트는 매 단계에 reference id를 넣고, reference id가 최고 logit과 0.0625 이내일 때만 다른 argmax를 허용합니다. 허용 횟수는 20단계 중 최대 2번입니다. 양방향 prefill(#991)은 logit 단위로 크게 벗어나므로 여전히 실패합니다.

## 7. 변경 요약

decode 처리량은 `mlxcel-bench-decode` tg128 중앙값 tok/s, M5 Max 기준입니다. 다른 unit의 컴파일과 호스트를 같이 써서 절대값은 #3129보다 10~18% 낮지만, 두 번의 sweep 모두 비율은 유지되었습니다.

| prompt | V2 기본값(absorbed) | V2 `=0` | 변화 | Kimi-VL fused | Kimi-VL fused 끔 | 변화 |
|---|---|---|---|---|---|---|
| 512 | 168.5 | 171.3 | -1.7% | 159.5 | 149.4 | +6.7% |
| 2048 | 153.5 | 150.9 | +1.8% | 149.8 | 139.2 | +7.5% |
| 16384 | 105.8 | 71.4 | +48% | | | |

teacher-forced trace (`--decided 2`, 너비 1):

| 비교 | 문맥 | decided 불일치 |
|---|---|---|
| V2-Lite, 기본값 대 `=0` | 1024 | 0 / 209 |
| V2-Lite, 기본값 대 `=0` | 4096 | 0 / 106 |
| Kimi-VL, fused 대 끔 | 1024 | 0 / 192 |
| Kimi-VL, fused 대 끔 | 4096 | 0 / 50 |

같은 decoder를 쓰는 deepseek-vl2-small-4bit에서는 두 레이아웃의 greedy 출력이 텍스트 프롬프트와 이미지 프롬프트 모두 동일합니다.

테스트: `the_default_absorbs_only_where_the_fused_kernel_runs`가 모드와 커널 사용 가능 여부의 여섯 조합을 모두 다룹니다. `mla::`와 `deepseek` lib 테스트가 통과하고, `deepseek_v2_greedy_parity`와 `deepseek_vl_v2_parity`는 기본값, `=0`, `=1`, `MLXCEL_MLA_FUSED_DECODE=0`에서 모두 통과합니다.

결과: `docs/benchmark_results/mla-absorbed-default-m5max-2026-11-16.md`.

## 8. 후속 조치

- #3133: CUDA·ROCm에서 기본값을 바꾸기 전에 absorbed와 decompressed를 비교 측정합니다.
- `src/execution/kv_arch.rs`는 아직 `deepseek_v2` KV 메모리를 decompressed 기준으로 계산해서, 새 Metal 기본값에서는 실제보다 크게 잡습니다. 예산 계산에는 안전하지만 정확한 추정치는 아닙니다.
