# PR #2418: iquestloop chunked prefill 허용 범위를 측정된 bf16 노이즈로 제한

**작성일**: 2026-10-16
**상태**: 실제 40B 체크포인트로 CUDA(GB10)에서 구현 및 검증 완료. 전체 `make verify-test-cuda` 워크스페이스 게이트와 Metal은 실행하지 않음
**위험도**: 낮음 (테스트 코드와 doc comment만 변경. 프로덕션 변경 없음)

## 요약

이슈 #2143은 GB10에서 `iquestloopcoder_matches_the_float32_oracle`가 `check_chunked_prefill`의 하드코딩된 0.25 logit 한계에서 실패한다고 보고했다 (`chunk 97: logit for token 5123 is 29.5, single-pass says 29.125`). 원인은 chunk 경계 로직이 아니라 bf16 activation 정밀도다. 모든 bf16 텐서를 로드 시점에 f32로 승격하면 chunked와 single-pass prefill이 76 800개 logit 전체 행에서 2.1e-05 이내로 일치한다. 이 PR은 0.25 한계를 측정된 bf16 한계로 교체하고, 경계 오류를 실제로 잡을 수 있는 검사는 새 f32 합성 테스트로 옮긴다.

## 문제 정의

chunk 97은 64 토큰 `loop_window_size`를 가로지르며, 이 지점에서 pass 2의 `RotatingKVCache`(`kept + L` 키)와 `mlxcel_core::causal_attention`이 만드는 윈도 마스크가 일치해야 한다. 크기 29에서 bf16 3 ulp인 차이는 이 경계의 off-by-one일 수도, 누적된 반올림일 수도 있었다. 이슈는 218 토큰 프롬프트에서 f32 oracle 대비 1.7 ~ 1.8 차이와 2, 3위 순위 뒤바뀜도 함께 언급했다.

## 근본 원인

두 경로는 같은 키, 오프셋, 마스크를 쓰고 bf16에서 반올림이 다를 뿐이다. GB10에서 `~/models/mlx/iquest-coder-v1-40b-loop-instruct-4bit`, `--test-threads=1`로 측정했고, 실행마다 결정적이며 `MLXCEL_SDPA_DETERMINISTIC=1` 유무와 결과가 같다. 모든 bf16 텐서 1445개를 로드 시 f32로 승격하는 임시 패치(커밋하지 않음)를 사용했다.

| 실행 | chunked vs single-pass, 전체 행 | mlxcel vs f32 oracle, 218 토큰 top-5 |
|---|---|---|
| bf16 (현재 배포) | chunk 128 / 97 / 64 / 32의 top-5 최악 0.000 / 0.500 / 0.375 / 0.625, 행 최대 1.23 | 1.65 ~ 1.87, 2, 3위 뒤바뀜 |
| f32, TF32 켬 | 행 최대 0.018 이하 | 약 0.037 |
| f32, `MLX_ENABLE_TF32=0` | 행 최대 2.1e-05 이하 (chunk 1: 5e-04) | 4.7e-04, oracle 순위 일치 (38 토큰 프롬프트: 1.3e-04) |

chunk 63과 65도 같은 결과다. 4-bit 가중치는 원인이 아니다 (f32 실행도 같은 가중치를 쓴다). oracle 차이와 순위 뒤바뀜도 bf16 activation에서 나온다. 어느 bf16 경로도 더 정확하지 않다: KL(f32 || bf16 single)은 0.0047이고 chunked 행은 0.0041 ~ 0.0071이다. chunk 크기에 따라 attention 호출 shape이 달라지고 (짧은 마스크 꼬리는 cuDNN을 벗어나 MLX의 비융합 fallback을 탈 수 있고, continuation 키는 다른 블록 오프셋에서 시작한다), shape마다 bf16 중간값의 반올림이 다르다.

## 기술적 선택과 그 이유

- `check_chunked_prefill`은 단언 전에 네 chunk 크기를 모두 측정하고 모든 실패를 보고한다. greedy 토큰(single-pass 최상위에서 `TIE_GAP` 이내의 토큰으로만 바뀔 수 있음)과 `TIE_GAP`으로 구분되는 순서는 유지하고, top-5 최악 차이는 1.0 (측정 최악 0.625), 전체 행 `KL(single || chunked)`는 0.015 (측정 최악 0.0074)로 제한한다. 측정 표는 doc comment에 기록했다.
- `src/models/iquestloopcoder_tests.rs`에 f32 합성 테스트 `chunked_prefill_matches_single_pass_past_the_window`를 추가했다: 218 토큰, 윈도 64, chunk 스케줄 128/97/65/64/63/32/1, 그리고 100, 단일 토큰 30개, 88 (링이 한 바퀴 돈 뒤의 다중 토큰 append). 모든 위치를 1e-4 한계와 비교한다 (정상 최대 5.4e-07). 기존 64+32 테스트는 마지막 행만 보므로 캐시가 키 하나를 놓치는 오류를 놓쳤다. continuation chunk가 유지하는 가장 오래된 키는 그 chunk의 첫 행에만 보인다.
- 두 검사 모두 먼저 모든 logit이 유한한지 단언한다. `f32::max`와 한계 비교는 NaN을 건너뛰는데, NaN이 이 경계의 과거 실패 형태이기 때문이다 (#401/#408).
- f16과 양자화된 `lm_head`를 원인으로 적었던 doc comment를 bf16으로 정정했다.
- 이슈 조사 코멘트가 제안한 수정(범위 지정 cuDNN dispatch, 정렬을 보존하는 K/V 패딩)은 논리 오류가 아니라 정밀도 격차를 줄이는 것이므로 후속으로 남긴다.

트레이드오프: bf16 한계는 어쩔 수 없이 거칠다. 게이트가 pass 2 local branch에 혼합의 약 12%만 주므로 pass 2 결함은 logit에서 작게 나타난다.

## 각 검사가 잡을 수 있는 것과 없는 것

연속 chunk의 `forward_pass2`에 주입한 결함:

| 결함 | 실제 체크포인트, bf16 | 새 f32 테스트 |
|---|---|---|
| 캐시가 키를 하나 덜 유지 | 통과 (KL 최대 0.0031) | 실패, 위치 128에서 6.4e-03 |
| pass 2 RoPE 오프셋 -1 | 통과 (KL 최대 0.0015) | 실패, 7.2e-02 |
| pass 2를 위치 0부터 회전 | KL로 실패 (0.024 ~ 0.030) | 실행 안 함 |

bf16 검사는 큰 결함만 잡는다. 경계 seam은 f32 테스트가 고정한다.

## 변경 사항

- `tests/iquestloopcoder_parity.rs` (+159, -35): `CHUNKED_LOGIT_TOLERANCE`, `CHUNKED_KL_TOLERANCE`, `kl_divergence`, `top_k_of_row`, 재작성한 `check_chunked_prefill`, f16에서 bf16으로의 doc 정정.
- `src/models/iquestloopcoder_tests.rs` (+95): 새 f32 테스트.

## 동작 변경

프로덕션 변경 없음. 테스트 코드와 주석만 바뀐다.

## 검증

- main의 기존 테스트는 이슈가 보고한 대로 실패한다. 새 테스트는 43초에 통과한다: `gpu-lock run -- target/test-fast/deps/iquestloopcoder_parity-<hash> --test-threads=1`, `models/mlx/...`가 해석되도록 `$HOME`에서 실행.
- `models::iquestloopcoder_tests` 18/18 통과. Clippy(`--lib --tests -D warnings`), fmt, 계약 검사 5개 통과.
- 결함 주입 결과는 위 표에 있다.

## 미검증 및 후속 조치

- Metal은 실행하지 않았고, 전체 `make verify-test-cuda` 워크스페이스 게이트는 머지 게이트에 맡긴다.
- 범위 지정 cuDNN dispatch와 정렬 보존 K/V 패딩은 bf16 chunked-single 격차를 줄일 수 있다. 선택적인 정밀도 작업이다.
