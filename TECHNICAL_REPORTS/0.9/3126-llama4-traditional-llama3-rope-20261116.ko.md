# 기술 보고서: PR #3126 - Llama 4에 traditional llama3 스케일 RoPE 적용

**작성일**: 2026-11-16
**상태**: 완료 (VLM 이미지 프롬프트와 멀티 노드 실행은 같은 attention을 쓰지만 실제로 돌려보지 않음)
**위험도**: 높음 (Llama 4의 모든 출력이 reference 쪽으로 바뀜)

## 요약

mlxcel의 Llama 4는 q와 k를 head를 반으로 나누는 RoPE(`traditional = false`)와 `rope_scale = 1.0`으로 회전시켰고, checkpoint의 `rope_scaling`(llama3, factor 16, original 8192)은 무시했다. mlx-lm은 `initialize_rope(..., traditional=True, scaling_config=rope_scaling)`를 쓴다. 그래서 Llama 4의 모든 위치가 잘못 회전되고 있었다(#3120, #3105를 trace하다가 발견). 새 `Llama4Rope`는 인접한 차원끼리 짝지어 회전하고, frequency table과 base는 기존 `RopeScalingKind::resolve`와 `rope_overrides::resolve_base`에서 가져온다. `CxxAttention`의 `fast_rope`/`fast_rope_batched` 호출 8곳이 모두 이것을 거친다. VLM loader와 pipeline stage도 같은 block을 만들기 때문에 함께 고쳐진다.

llama-4-scout를 기본 설정의 mlx-lm 0.31.3과 비교하면, decided position 불일치가 prefill, decode, B=2 batched decode 모두에서 83~92%에서 0.00~0.05%로 줄었다. perplexity는 1,800~3,000에서 mlx-lm의 10~16과 0.1% 이내로 같아졌다. greedy 출력은 같은 말을 반복하던 상태에서 정상적인 문장이 됐고, decode 속도는 그대로다.

## 3. 기술적 선택과 그 이유

- **기존 llama3 scaling을 재사용한다.** frequency table은 llama3와 helium이 이미 쓰는 것과 같아서 새로 만들지 않았다. 그 결과 `--rope-*` override도 이제 Llama 4에 적용되며, `rope_overrides.rs`, `rope_utils.rs`, `docs/llama-server-compat.md`의 지원 계열 목록에도 반영했다.

## 7. 변경 요약

| shape | decided position | main | 브랜치 | ppl mlx-lm / main / 브랜치 |
|---|---|---|---|---|
| prefill width 1280 | 1965 | 83.82% | 0.05% | 14.213 / 1868.2 / 14.217 |
| prefill width 4608, 문맥 최대 9216 | 3643 | 87.90% | 0.00% | 11.803 / 2392.9 / 11.800 |
| decode width 1, 문맥 512~2560 | 750 | 83.33% | 0.00% | 16.351 / 1819.6 / 16.349 |
| batched decode B=2 | 750 | 83.47% | 0.00% | 16.351 / 1815.7 / 16.367 |
| decode width 1, 문맥 4608~9216 | 1886 | 92.42% | 0.00% | 10.268 / 3059.5 / 10.260 |

테스트:
- 새 테스트 두 개는 single-row 회전과 batched 회전을 mlx-lm의 traditional `Llama3RoPE`를 f64로 옮긴 reference와 offset 20000까지 비교한다. 예전 회전 방식은 이 reference와 맞지 않는다는 것도 함께 확인한다.
- 전체 `mlxcel` lib suite(11,986개)가 통과했고, workspace clippy도 경고가 없다.

## 8. 후속 조치

main에서 폭 9216짜리 Llama 4 forward 한 번이 Metal out-of-memory로 중단되어, 측정에서는 폭 4608 forward 두 번으로 나눠 돌렸다. 원인은 아직 조사하지 않았다.
