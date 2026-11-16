# 기술 보고서: PR #3118 - 측정 결과에 따라 DeepSeek V2 기본 MLA cache를 decompressed로 유지

**작성일**: 2026-11-16
**상태**: 완료 (결정을 기록했고 모델 코드는 바꾸지 않음)
**위험도**: 낮음

## 요약

#3102는 DeepSeek V2의 기본값을 absorbed(latent) MLA cache로 바꿀지 판단하는 이슈였다. absorbed cache는 28K 토큰을 2048 토큰 청크로 처리할 때 피크가 약 6.7 GB 낮다. 기본값을 바꾸려면 decode 처리량 저하가 몇 % 이내여야 하고, trace에서도 의미 있는 불일치가 없어야 했다. trace 조건은 통과했다. 네 가지 shape에서 decided position 3,653개 중 불일치가 0개였고, perplexity 변화는 최대 0.03%였다. 처리량 조건은 통과하지 못했다. absorbed decode는 pp512/tg128에서 9.9% 느렸고(180.7 대 200.6 tok/s), 2048 토큰 프롬프트 뒤에서는 3.7% 느렸으며, 16K 프롬프트 뒤에서는 57% 빨랐다. 그래서 decompressed 구조를 기본값으로 유지하고, `MLXCEL_MLA_ABSORBED=1`은 계속 선택 사항으로 둔다. 측정값과 판단을 바꿀 조건은 `docs/mla-absorbed-decode.md`에 기록했다.

## 3. 기술적 선택과 그 이유

- **판단 기준은 짧은 문맥의 decode다.** 대부분의 배포 환경이 이 조건에 해당하고, 이슈도 기준을 여기에 두었다. 두 방식의 처리량이 뒤집히는 지점은 문맥 약 3K 토큰이다. absorbed decode에는 단계마다 약 0.55 ms의 고정 비용이 있으며, 후속 이슈에서 이 원인을 추적할 수 있다. 후보는 dense f16으로 접은 `W_UK`/`W_UV` contraction과 조합형 attention이다.
- **다른 모델에는 영향이 없다.** 이 스위치를 읽는 모델은 `deepseek_v2`뿐이다. V3와 V3.2는 항상 absorbed 경로를 쓰고, minicpm3와 dots1에는 absorbed 경로가 없다.
- **측정 도구 수정.** `examples/logit_trace.rs`는 MLA 경로 카운터를 한쪽 경로에서만 출력해서, 폭 1 decode trace가 어느 경로를 탔는지 알 수 없었다. 이제 두 경로 모두 카운터를 출력한다.

## 7. 변경 요약

| 측정 | decompressed (기본값) | absorbed |
|---|---|---|
| decode pp512/tg128, 중앙값 | 200.6 tok/s | 180.7 tok/s |
| 16,384 토큰 뒤의 decode | 70.9 tok/s | 111.2 tok/s |
| 28,705 토큰 피크, 2048 청크 | 20.20 GB | 13.52 GB |
| 피크, 단일 패스 | 48.55 GB | 41.02 GB |

자세한 내용: `docs/benchmark_results/mla-v2-default-layout-m5max-2026-11-15.md`.
