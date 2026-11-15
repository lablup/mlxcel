# 기술 보고서: PR #3087 - Mamba2 SSD를 청크로 나누고 Mamba1 graph scan을 블록 단위로 처리

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 중간

## 요약

Falcon-H1, GraniteMoeHybrid, PLaMo 2는 Mamba2 SSD를 prefill piece 전체에 대해 한 번에 계산했기 때문에 층마다 f32 `[B, H, L, L]` segsum 사본을 여러 개 들고 있었다. 그 결과 M5 Max에서 granite-4.0-h-tiny, falcon-h1-tiny, plamo-2-1b의 28K 단일 패스는 메모리가 부족해 실패했다. 이제 세 계열 모두 공용 `chunked_ssd_scan`을 256 토큰 청크로 호출한다. 단일 패스가 끝까지 돌아가고(17.10, 9.99, 25.62 GB), 2048 토큰 청크의 피크가 줄었으며(9.85→7.46, 3.99→2.34, 12.21→9.90 GB), prefill은 1.6~2.7배 빨라졌다. Jamba의 graph-fallback selective scan은 이제 state 증분을 piece 전체가 아니라 256 단계 블록마다 만든다.

## 1. 문제 정의

falcon_h1과 granitemoehybrid의 `ssm_step`, 그리고 plamo2의 scan은 청크로 나누지 않은 SSD 방식을 썼다. 이 방식의 작업 메모리는 piece 길이의 제곱에 비례해 커지며, head 128개에 L = 2048이면 사본 하나가 약 2 GiB다. `mamba2::ssm_attn`과 nemotron_h는 이미 청크로 나눠 계산하고 있었다(#3073).

## 3. 기술적 선택과 그 이유

- **scan은 하나로 공유한다.** 세 곳에 중복돼 있던 `segsum`/`repeat_axis` 구현을 `src/models/ssd_scan.rs` 하나로 대체했다. NemotronH의 `nemotron_h_ssm_attn`은 이 함수를 청크마다 평가하는 얇은 wrapper가 되어 동작이 그대로다. 청크 크기 256은 mlx-lm `ssm_attn`의 기본값이며 prefill piece 크기와 무관하다.
- **#3063 동작은 그대로 유지한다.** 최종 state는 여전히 `schedule_prefill_state`로 예약하고, conv tail은 `materialized_state_tail`을 쓴다. 2048 토큰보다 긴 단일 패스에서는 각 window의 출력과 state도 예약한다.
- **연산 순서가 바뀐다.** 청크 scan은 piece 전체 scan과 합산 순서가 다르다. 모델당 16,896개 위치(forward width 2112)에서 teacher-forced logit trace로 본 decided position 불일치는 다음과 같다.
  - granite-4.0-h-tiny: 16,148개 중 4개(0.025%)
  - falcon-h1-tiny: 12,168개 중 0개
  - plamo-2-1b: 9,400개 중 0개

  perplexity 변화는 -0.098%~+0.007%다. 2048 청크에서 Granite와 PLaMo 2의 greedy 출력은 바이트 단위로 동일하다. Falcon-H1 tiny는 양쪽 모두 이미 반복적인 출력에서 갈라진다.
- **`mamba.rs`는 바꾸지 않는다.** graph fallback이 이미 시간 단계 하나씩 projection하므로, 이슈가 지적한 `[B, L, d_inner, N]` 버퍼가 여기에는 없다.

## 7. 변경 요약

| Checkpoint | 단일 패스, main → 브랜치 | 2048 청크 피크, main → 브랜치 | 2048 청크 prefill |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | OOM → 17.10 GB | 9.85 → 7.46 GB | 1.6~2.5배 빠름 |
| falcon-h1-tiny-90m-instruct-4bit | OOM → 9.99 GB | 3.99 → 2.34 GB | 2.6~2.7배 빠름 |
| plamo-2-1b | OOM → 25.62 GB | 12.21 → 9.90 GB | 1.5~1.6배 빠름 |
| ai21-jamba-reasoning-3b-4bit (대조군, Metal에서 fused kernel 사용) | 6.45 → 6.45 GB | 3.25 → 3.25 GB | 변화 없음 |

새 테스트:
- 청크 scan과 비청크 scan을 f32와 f16에서, 이전 state가 있을 때와 없을 때 비교한다.
- 평가 정책을 바꿔도 결과가 비트 단위로 같은지 확인한다.
- Jamba의 블록 단위 graph scan이 비트 단위로 같은지 확인하고, fallback을 강제한 prefill을 kernel 결과와 비교한다.

계열별 unit test와 `tests/mamba2_hybrid_*`가 통과한다. 자세한 내용은 `docs/benchmark_results/ssm-prefill-scan-chunking-m5max-2026-11-15.md`에 있다.

## 8. 후속 조치

- GraniteMoeHybrid fused-MoE 테스트 두 개(`granite_hybrid_fused_kernel_matches_the_gather_path_at_one_token`, `granite_hybrid_fused_arm_routes_one_token_through_the_fused_kernel`)가 병렬 실행에서는 실패하고 단독 실행에서는 통과한다. 이 PR은 해당 경로를 건드리지 않는다.
