# PR #3332: LongCat-Flash 잔여 차이의 원인을 mlx-lm의 kv_b_proj 재양자화로 규명

**날짜**: 2026-11-23
**상태**: 드래프트 PR (#3318 종료). 문서와 스크립트 docstring만 바꾼다. 실제 체크포인트 비교 결과는 아래에 있다.
**위험도**: 낮음 (런타임 코드 변경 없음)

## 요약

#3306 이후에도 `longcat-flash-lite-4bit`을 mlx-lm과 teacher-forced trace로 비교하면 36864개 위치에서 decided 불일치가 20486개 중 39개 남아 있었다. 이 잔여 차이는 두 런타임이 서로 다른 가중치를 읽어서 생긴다. mlxcel 코드가 어긋나서 생긴 것이 아니다. 체크포인트는 융합된 4비트 `kv_b_proj`를 담고 있다. mlxcel은 이를 dense `embed_q` / `unembed_out`으로 나누고, mlx-lm은 나눈 두 절반을 다시 양자화한다. reference가 나눈 가중치를 dense로 두면 잔여 차이는 4개로 줄어든다. dtype 노이즈 바닥(bf16 reference와 같은 reference를 float16으로 바꾼 것의 차이)은 5개다.

## 1. 문제

#3295의 첫 번째 수용 기준은 reference 자체의 노이즈를 넘는 decided 불일치가 없을 것을 요구했다. #3306 이후에는 39개가 남았다. 청크당 개수는 12288 이전의 4096 위치 청크마다 0~1개에서 12288 이후 청크마다 4~10개로 늘어났다. 같은 `MlaRope`를 쓰는 GigaChat3는 0개였으므로, 이 차이는 LongCat에만 있는 것으로 보였다.

## 2. 변경 요약

| 영역 | 변경 |
| --- | --- |
| `scripts/mlx_lm_logit_trace.py` | docstring에서 두 쪽이 모두 체크포인트 dtype으로 돈다는 주장을 뺐다. mlxcel은 Apple Silicon에서 양자화되지 않은 체크포인트만 bf16을 f16으로 바꾼다. 융합 양자화 `kv_b_proj`는 두 런타임이 서로 다른 가중치로 읽는다는 점도 적었다. |
| `docs/supported-models.md` | LongCat 항목의 "원인 미확인"을 실제 원인으로 바꿨다. |
| `docs/benchmark_results/longcat-flash-lite-trace-residual-m3ultra-2026-11-23.md` | 모든 비교 조건의 청크별 결과. |

## 3. 기술적 결정

### 3.1 이슈가 정한 순서대로 비교

1. `MLXCEL_KEEP_BF16=1`: 행이 바이트 단위로 같았다. `bf16_to_f16_at_load`는 양자화 체크포인트에 false를 돌려주므로, 이 체크포인트는 처음부터 변환되지 않았다. 이슈의 전제가 맞지 않았다.
2. `MLXCEL_TRACE_KV_CACHE_MODE=fp16`: 행이 같았다. dense 캐시는 mlx-lm `KVCache`처럼 이미 bf16 activation을 저장한다.
3. `MLXCEL_TRACE_MLA_STATS=1`: 모든 step이 materializing 경로였으므로 강제로 바꿀 absorbed 경로가 없었다. 로컬 실험으로 mlx-lm의 방식을 되살렸다. MLX가 비융합으로 실행하는 192폭 folded SDPA 대신, rope 항을 bf16 additive mask로 두고 128폭 fused SDPA를 썼다. 결과는 33개였다.

### 3.2 레이어별 이분 탐색 없이 원인을 찾음

float32 mlx-lm reference는 float16 reference와 decided 불일치 0개로 일치했다. 그래도 mlxcel과는 28개 차이가 났고, 청크 1~3에서 mlxcel의 perplexity가 5~7% 낮았다. 이만큼 일관된 차이는 가중치가 다르다는 신호였다. 두 sanitize를 비교해 `kv_b_proj` 재양자화를 찾았다. 나눈 가중치를 dense로 두는 reference shim을 쓰면 불일치가 4개(bf16), 2개(float32)로 줄었다.

### 3.3 mlxcel의 분할 방식은 유지

mlxcel의 dense 절반은 체크포인트를 역양자화한 가중치와 정확히 같다. mlx-lm은 `embed_q`를 전치된 축을 따라 한 번 더 양자화하면서 정밀도를 잃는다. 여기에 맞추면 mlxcel이 체크포인트를 덜 충실하게 재현하게 된다.

## 4. 검증

| Reference | Candidate | Decided | 청크별 |
| --- | --- | --- | --- |
| bf16 stock | mlxcel main | 39 / 20486 | 0 1 0 4 10 4 6 8 6 |
| bf16 stock | float16 reference | 5 / 20486 | 0 0 0 1 1 0 0 1 2 |
| bf16 dense split | mlxcel main | 4 / 20489 | 0 0 0 2 1 0 0 0 1 |
| float32 dense split | mlxcel main | 2 / 20208 | 0 1 0 1 0 0 0 0 0 |
| bf16 dense split | bf16 stock | 32 / 20489 | 0 3 0 5 4 3 4 7 6 |

계약 검사는 통과했다. fmt, `scripts/ci` 스크립트, 라이선스 헤더 검사가 여기에 들어간다. `coverage_tests`와 `longcat` lib selector도 통과했다.

## 5. 미검증 및 남은 항목

- dense split과 캐시 prefill을 쓰는 reference는 로컬 shim이다. 저장소의 스크립트는 여전히 기존 분할 방식으로 한 번의 forward를 실행한다.
- sparse 변형도 HF 레이아웃에서 같은 종류의 융합 `kv_b_proj`를 읽는다. 그 mlx-vlm reference가 재양자화를 하는지는 여기서 확인하지 않았다. sparse 변형의 trace는 이미 0개였다.

## 6. 배운 점

- 한쪽이 내용과 관계없이 f32 오라클보다 일관되게 확신이 높다면, 연산을 이분 탐색하기 전에 각 런타임이 무엇을 읽는지부터 비교한다.
- dtype만 바꾼 reference 쌍은 실제 노이즈 바닥을 준다. #3306의 prefill step 쌍은 0개여서 노이즈 바닥을 정하지 못했다.

## 관련 커밋

- `cb8a03b5` docs: trace LongCat-Flash residual to mlx-lm's kv_b_proj requantization
